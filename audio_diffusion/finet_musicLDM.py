import torch, random
import numpy as np
import torch.nn.functional as F
from tqdm.auto import tqdm
from IPython.display import Audio
from matplotlib import pyplot as plt
from diffusers import DiffusionPipeline, AudioPipelineOutput
from torchaudio import transforms as AT
from torchvision import transforms as IT
import torchaudio
import transformers
from transformers import ClapProcessor, ClapModel, AutoProcessor
from torch.optim import AdamW, Adam
from torch.utils.data import DataLoader, Dataset
import os
from diffusers import MusicLDMPipeline
import torch.nn.functional as F
from librosa.filters import mel as librosa_mel_fn
import sys
# from torch.nn import DataParallel
# Add the root directory to the Python path
sys.path.append(os.path.abspath(".."))
from data.audioLDM_pre import *

os.environ["TOKENIZERS_PARALLELISM"] = "false"

text_tensor =  np.load('/srv/nfs-data/sisko/matteoc/music/music_data_caps_capt.npy').tolist()
audio_tensor = torch.load('/srv/nfs-data/sisko/matteoc/music/music_data_caps_audio.pt')

repo_id = "ucsd-reach/musicldm"
musicldm_pipe = MusicLDMPipeline.from_pretrained(repo_id, torch_dtype=torch.float32)
device = "cuda:3" if torch.cuda.is_available() else "cpu"
musicldm_pipe = musicldm_pipe.to(device)

clap_model_id = "laion/larger_clap_music_and_speech"
clap_model = ClapModel.from_pretrained(clap_model_id).to(device)
clap_process = AutoProcessor.from_pretrained(clap_model_id)

sampling_rate_ldm = 16000
n_mel_channels = 64
mel_fmin = 0
mel_fmax = 8000
duration = 10.0
filter_length = 1024
hop_length = 160
win_length = 1024 
window = 'hann'
target_length = int(duration * sampling_rate_ldm / hop_length)
pad_wav_start_sample = 0

stft = STFT(
    filter_length=filter_length, 
    hop_length=hop_length, 
    win_length=win_length,
    window=window
)

class AudioDataset(Dataset):
    def __init__(self, dataset, captions=None, sample_rate_dataset=44100, new_sr=sampling_rate_ldm):
        self.dataset = dataset
        self.captions = captions
        self.resampler = torchaudio.transforms.Resample(orig_freq=sample_rate_dataset, new_freq=new_sr)
        # vself.resampler_clap = torchaudio.transforms.Resample(orig_freq=sample_rate_dataset, new_freq=48000)

    def __len__(self):
        return len(self.dataset)

    def __getitem__(self, idx):
        audio_array = self.dataset[idx]
        audio_res = self.resampler(audio_array)

        if self.captions is not None:
            caption = self.captions[idx]
            return audio_res, caption
         
        else:
            return audio_res
    
    
def get_mel_features(audio):
    magnitude, phase = stft.transform(audio)
    mel_basis = librosa_mel_fn(
                sr=sampling_rate_ldm, n_fft=filter_length, n_mels=n_mel_channels, fmin=mel_fmin, fmax=mel_fmax
            )
    mel_basis = torch.from_numpy(mel_basis).float()
    magnitudes = magnitude.data
    mel_output = torch.matmul(mel_basis, magnitudes)
    mel_output = spectral_normalize(mel_output, torch.log).permute(0,2,1)

    return mel_output
    
seed = 55
torch.manual_seed(seed)
random.seed(seed)
    
audio_dataset = AudioDataset(audio_tensor, text_tensor)

train_size = int(0.8 * len(audio_dataset))
val_size = len(audio_dataset) - train_size
train_dataset, val_dataset = torch.utils.data.random_split(audio_dataset, [train_size, val_size])

train_dataloader = DataLoader(train_dataset, batch_size=16, shuffle=True, num_workers=4)
val_dataloader = DataLoader(val_dataset, batch_size=4, shuffle=False, num_workers=4)

for step, batch in (enumerate(val_dataloader)):
    batch_sample = 0
    batch_audio = batch[0][batch_sample].unsqueeze(0)
    batch_audio_zeros = torch.zeros(1, 160000)
    print('batch_audio: ', batch_audio.shape)
    batch_text = batch[1][batch_sample]
    real_audio_val = get_mel_features(batch_audio).to(device).unsqueeze(1)
    audio_val_zeros = get_mel_features(batch_audio_zeros).to(device).unsqueeze(1)
    real_text_val = batch_text
    del batch
    print('real_audio_val: ', real_audio_val.shape)
    print('real_text_raw: ', real_text_val)
    audio_features_val = clap_model.get_audio_features(real_audio_val)
    audio_val_zeros = clap_model.get_audio_features(audio_val_zeros)
    audio_features_val = torch.cat((audio_features_val, audio_val_zeros), dim=0)
    prompt_embeds_val = musicldm_pipe._encode_prompt(
            real_text_val,
            device,
            num_waveforms_per_prompt=1,
            do_classifier_free_guidance=2.0,
            negative_prompt='',
            prompt_embeds=None,
            negative_prompt_embeds=None,
        )
    print('audio_features_val: ', audio_features_val.shape)
    print('prompt_embeds_val: ', prompt_embeds_val.shape)
    break


model = musicldm_pipe.unet

output_type = "np"
audio_length_in_s = 10.0
num_inference_steps = 50
cross_attention_kwargs = None
guidance_scale = 5.0     # TODO: sto aggiornando
do_classifier_free_guidance = guidance_scale > 1.0
callback = None
callback_steps = 1
extra_step_kwargs= {}
extra_step_kwargs["eta"] = 0.0
extra_step_kwargs["generator"] = torch.Generator(device=device).manual_seed(55)

num_epochs = 10  
lr = 1e-4 
grad_accumulation_steps = 2  

optimizer = torch.optim.AdamW(model.parameters(), lr=lr)
train_losses = []
val_losses = []

import wandb 
wandb.login(key='41a4723fac40aff96b88423b6d3e15dd64f87488')
wandb.init()


def training_step(batch_audio, batch_text, bs):

    batch_mel = get_mel_features(batch_audio).to(device).unsqueeze(1)
    encoded = musicldm_pipe.vae.encode(batch_mel)
    latents_real = musicldm_pipe.vae.config.scaling_factor * encoded.latent_dist.mean
    audio_features = clap_model.get_audio_features(batch_mel)
    text_features = musicldm_pipe._encode_prompt(
        list(batch_text),
        device,
        num_waveforms_per_prompt=1,
        do_classifier_free_guidance=False,
        negative_prompt="Low quality or muffled sound",
        prompt_embeds=None,
        negative_prompt_embeds=None,
    )

    # noise = torch.randn(latents_real.shape).to(latents_real.device)
    noise = musicldm_pipe.prepare_latents(
        bs,  #  --> da moltiplicare se num_waveforms_per_prompt > 1
        musicldm_pipe.unet.config.in_channels,
        1000,  # height
        torch.float32,
        torch.device(device),
        generator=torch.Generator(device=device).manual_seed(42),
        latents=None,
    )

    return latents_real, audio_features, text_features, noise

    
def inference_musicldm(pipeline, model_ft, latents, features_val, device):

    do_classifier_free_guidance = guidance_scale  # Modify Here
    pipeline.scheduler.set_timesteps(num_inference_steps=num_inference_steps, device=device)
    timesteps = pipeline.scheduler.timesteps
    num_warmup_steps = len(timesteps) - num_inference_steps * pipeline.scheduler.order
    with pipeline.progress_bar(total=num_inference_steps) as progress_bar:
        for i, t in enumerate(timesteps):
            # expand the latents if we are doing classifier free guidance
            latent_model_input = torch.cat([latents] * 2) if do_classifier_free_guidance else latents
            latent_model_input = pipeline.scheduler.scale_model_input(latent_model_input, t)

            # predict the noise residual
            noise_pred_eval = model_ft(
                latent_model_input,
                t,
                encoder_hidden_states=None,
                class_labels=features_val,
                cross_attention_kwargs=cross_attention_kwargs,
                return_dict=False,
            )[0]

            # perform guidance
            if do_classifier_free_guidance:
                noise_pred_uncond, noise_pred_text = noise_pred_eval.chunk(2)
                noise_pred_eval = noise_pred_uncond + guidance_scale * (noise_pred_text - noise_pred_uncond)

            # compute the previous noisy sample x_t -> x_t-1
            latents = pipeline.scheduler.step(noise_pred_eval, t, latents, **extra_step_kwargs).prev_sample

            # call the callback, if provided
            if i == len(timesteps) - 1 or ((i + 1) > num_warmup_steps and (i + 1) % pipeline.scheduler.order == 0):
                progress_bar.update()
                if callback is not None and i % callback_steps == 0:
                    step_idx = i // getattr(pipeline.scheduler, "order", 1)
                    callback(step_idx, t, latents)
   
    if not output_type == "latent":
        latents = 1 / pipeline.vae.config.scaling_factor * latents
        print('LATENT HAT MEAN: ', latents.mean())
        print('LATENT HAT STD: ', latents.std()) 
        mel_spectrogram = pipeline.vae.decode(latents).sample
    
    original_waveform_length = int(audio_length_in_s * pipeline.vocoder.config.sampling_rate)
    audio_to_save = pipeline.mel_spectrogram_to_waveform(mel_spectrogram.to(device=device))
    audio_to_save = audio_to_save[:, :original_waveform_length]

    if output_type == "np":
        audio = audio_to_save.detach().numpy()
    audio_pipe = AudioPipelineOutput(audios=audio)

    return audio_pipe


for epoch in range(num_epochs):

    model.train()
    for step, batch_train in tqdm(enumerate(train_dataloader), total=len(train_dataloader)):

        # batch = batch.to(device)
        bs = batch_train[0].shape[0]
        batch_audio = batch_train[0]
        batch_text = batch_train[1]
        latents_real, audio_features, _, noise = training_step(batch_audio, batch_text, bs)
        del batch_train
        
        # Sample a random timestep for each image
        timesteps = torch.randint(0, musicldm_pipe.scheduler.num_train_timesteps, (bs,), device=latents_real.device,).long()

        # Add noise to the clean images according to the noise magnitude at each timestep
        # (this is the forward diffusion process) --> we are in the training!
        noisy_latents = musicldm_pipe.scheduler.add_noise(latents_real, noise, timesteps)
        # optimizer.zero_grad()

        # Get the model prediction for the noise
        noise_pred = model(
            noisy_latents,
            timesteps,
            encoder_hidden_states=None,
            class_labels=audio_features,
            cross_attention_kwargs=None,
            return_dict=False,
        )[0]
        
        # Compare the prediction with the actual noise:
        loss = F.mse_loss(
            noise_pred, noise
        )  

        # Store for later plotting
        train_losses.append(loss.item())

        # Update the model parameters with the optimizer based on this loss
        loss.backward()
        # optimizer.step()
        wandb.log({'train_loss_step': loss.item()})

        # Gradient accumulation:
        if (step + 1) % grad_accumulation_steps == 0:
            optimizer.step()
            optimizer.zero_grad()

    print(f"Epoch {epoch} average loss: {sum(train_losses[-len(train_dataloader):])/len(train_dataloader)}")
    wandb.log({'average_train_loss': sum(train_losses[-len(train_dataloader):])/len(train_dataloader)})

    model.eval()
    for step, batch_val in tqdm(enumerate(val_dataloader), total=len(val_dataloader)):

        bs = batch_val[0].shape[0]
        batch_audio = batch_val[0]
        batch_text = batch_val[1]
        if step == 0:
            to_save_audio = batch_audio[0]
            to_save_text = batch_text[0]

        latents_real, audio_features, _, noise = training_step(batch_audio, batch_text, bs)
        del batch_val

        # Sample a random timestep for each image
        timesteps = torch.randint(0, musicldm_pipe.scheduler.num_train_timesteps, (bs,), device=latents_real.device,).long()

        # Add noise to the clean images according to the noise magnitude at each timestep
        # (this is the forward diffusion process) --> we are in the training!
        noisy_latents = musicldm_pipe.scheduler.add_noise(latents_real, noise, timesteps)

        # Get the model prediction for the noise
        noise_pred = model(
            noisy_latents,
            timesteps,
            encoder_hidden_states=None,
            class_labels=audio_features,
            cross_attention_kwargs=None,
            return_dict=False,
        )[0]
        
        # Compare the prediction with the actual noise:
        loss = F.mse_loss(
            noise_pred, noise
        )  

        # Store for later plotting
        val_losses.append(loss.item())
        wandb.log({'val_loss_step': loss.item()})

    print('LATENT MEAN: ', latents_real[0].mean())
    print('LATENT STD: ', latents_real[0].std()) 
    wandb.log({'average_val_loss': sum(val_losses[-len(val_dataloader):])/len(val_dataloader)})

    # device_inf = "cuda:5" if torch.cuda.is_available() else "cpu"
    audio_reconstr = inference_musicldm(musicldm_pipe, model, noise[0:1], audio_features_val, device)[0]

    wandb.log({"audio_reconstr": wandb.Audio(audio_reconstr.squeeze(), sample_rate=16000, caption=to_save_text)})
    wandb.log({"audio_real": wandb.Audio(to_save_audio, sample_rate=16000, caption=to_save_text)})



