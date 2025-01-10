import torch, random
import soundfile as sf
from diffusers import StableAudioPipeline
import numpy as np
import os
from transformers import ClapProcessor, ClapModel, AutoProcessor, ClapAudioModel, AutoFeatureExtractor
from diffusers import AudioPipelineOutput
from IPython.display import Audio
from torch.utils.data import DataLoader, Dataset

os.environ["TOKENIZERS_PARALLELISM"] = "false"

text_tensor =  np.load('/srv/nfs-data/sisko/matteoc/music/music_data_caps_capt.npy').tolist()
audio_tensor = torch.load('/srv/nfs-data/sisko/matteoc/music/music_data_caps_audio.pt')
sample_rate = 44100

stable_pipe = StableAudioPipeline.from_pretrained("stabilityai/stable-audio-open-1.0", torch_dtype=torch.float16)
device = 'cuda:1'
stable_pipe = stable_pipe.to(device)
clap_model_id = "laion/clap-htsat-unfused"
clap_model = ClapAudioModel.from_pretrained(clap_model_id).to(device)
processor = AutoProcessor.from_pretrained(clap_model_id)

negative_prompt = "Low quality."
generator = torch.Generator((device)).manual_seed(0)
do_classifier_free_guidance=7.5
negative_prompt_embeds=None
audio_start_in_s=0
audio_end_in_s=10.0

seconds_start_hidden_states, seconds_end_hidden_states = stable_pipe.encode_duration(
                audio_start_in_s,
                audio_end_in_s,
                device,
                do_classifier_free_guidance and (negative_prompt is not None or negative_prompt_embeds is not None),
                1,
            )

def get_clap_audio_embd(audio_tensor):
    inputs = processor(audios=audio_tensor, return_tensors="pt")
    outputs = clap_model(**inputs.to(device))
    last_hidden_state = outputs.last_hidden_state
    last_hidden_state = torch.mean(last_hidden_state, dim=2)
    last_hidden_state = last_hidden_state.permute(0,2,1)
    
    return last_hidden_state

def get_stable_text_embd(prompt):
    prompt_embeds = stable_pipe.encode_prompt(
            prompt,
            device=device,
            do_classifier_free_guidance=do_classifier_free_guidance,
            negative_prompt=negative_prompt,
            prompt_embeds=None,
            negative_prompt_embeds=negative_prompt_embeds,
            attention_mask=None,
            negative_attention_mask=None,
        )
    text_audio_duration_embeds = torch.cat(
                [prompt_embeds, seconds_start_hidden_states, seconds_end_hidden_states], dim=1
            )
    
    return text_audio_duration_embeds

seed = 42
torch.manual_seed(seed)
random.seed(seed)

class AudioDataset(Dataset):
    def __init__(self, dataset, captions=None):
        self.dataset = dataset
        self.captions = captions

    def __len__(self):
        return len(self.dataset)

    def __getitem__(self, idx):
        audio_array = self.dataset[idx]
        # audio_stereo = torch.cat((audio_array.unsqueeze(0), audio_array.unsqueeze(0)), dim=0)

        if self.captions is not None:
            caption = self.captions[idx]
            return audio_array, caption
         
        else:
            return audio_array
        
audio_dataset = AudioDataset(audio_tensor, text_tensor)

train_size = int(0.8 * len(audio_dataset))
val_size = len(audio_dataset) - train_size
train_dataset, val_dataset = torch.utils.data.random_split(audio_dataset, [train_size, val_size])

train_dataloader = DataLoader(train_dataset, batch_size=16, shuffle=True, num_workers=4)
val_dataloader = DataLoader(val_dataset, batch_size=4, shuffle=False, num_workers=4)

for step, batch in (enumerate(val_dataloader)):
    batch_sample = 0
    batch_audio = batch[0][batch_sample]
    batch_audio_zeros = torch.zeros(sample_rate)
    print('batch_audio: ', batch_audio.shape)
    batch_text = batch[1][batch_sample]
    del batch
    print('batch_text: ', batch_text)
    audio_features_val = get_clap_audio_embd(batch_audio)
    audio_val_zeros = get_clap_audio_embd(batch_audio_zeros)
    audio_features_val = torch.cat((audio_val_zeros, audio_features_val), dim=0)
    audio_features_val = torch.cat(
                [audio_features_val, seconds_start_hidden_states, seconds_end_hidden_states], dim=1
            )
    print('audio_features_val: ', audio_features_val.shape)
    prompt_embeds_val = get_stable_text_embd(batch_text)
    print('prompt_embeds_val: ', prompt_embeds_val.shape)
    break


        





