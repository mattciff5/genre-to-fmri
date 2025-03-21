import numpy as np
import scipy
import scipy.io as sio
import matplotlib.pyplot as plt
import nibabel as nib
import os
from os.path import join as opj
from os.path import join, exists, split
import warnings
from pprint import pprint
import zipfile
import glob
warnings.filterwarnings('ignore')
from transformers import AutoFeatureExtractor, ClapModel
import torch
import torchaudio
from sklearn.linear_model import LogisticRegression, RidgeCV, Ridge
import pandas as pd
from nilearn import maskers
import tqdm
from nilearn.glm.first_level import FirstLevelModel
from nilearn.image import concat_imgs, mean_img
import matplotlib.pyplot as plt
import nilearn
from nilearn.plotting import plot_design_matrix
from nilearn.plotting import plot_contrast_matrix
from importlib import reload # python 2.7 does not require this
from data_agg import *
import pickle
# from nltools.data import Brain_Data, Adjacency
# from nltools.stats import align
import seaborn as sns
import IPython.display as ipd
from sklearn.metrics import confusion_matrix
from diffusers import MusicLDMPipeline, AudioPipelineOutput, StableAudioPipeline
# from diffusers import DiffusionPipeline, AudioPipelineOutput
from IPython.display import Audio
os.environ["TOKENIZERS_PARALLELISM"] = "false"
from transformers import AutoFeatureExtractor, ClapModel, ClapProcessor
import torch.optim as optim
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler, normalize
import librosa
from IPython.display import Audio, display
import librosa
from joblib import load
import random
from scipy.io.wavfile import write


def save_spect(audio_to_save, file_name, path_to_save_img, flag):
    X_vera = librosa.stft(audio_to_save)
    Xdb_vera = librosa.amplitude_to_db(abs(X_vera), ref=np.max)
    plt.figure(figsize=(10, 4))
    librosa.display.specshow(Xdb_vera, sr=16000, x_axis='time', y_axis='hz',)
    plt.colorbar()
    plt.title('Spectrogram')
    plt.xlabel('Time (s)')
    plt.ylabel('Frequency (Hz)')
    plt.savefig(os.path.join(path_to_save_img, f'{file_name}_{flag}.png'))
    # plt.show()


repo_id = "ucsd-reach/musicldm"
musicldm_pipe = MusicLDMPipeline.from_pretrained(repo_id, torch_dtype=torch.float32)
device = "cuda:2" if torch.cuda.is_available() else "cpu"
musicldm_pipe = musicldm_pipe.to(device)

clap_model_id="laion/larger_clap_music_and_speech"
model = ClapModel.from_pretrained(clap_model_id).to(device)
processor = ClapProcessor.from_pretrained(clap_model_id)
feature_extractor = AutoFeatureExtractor.from_pretrained(clap_model_id)

with open('/data01/data/fMRI_music_genre/data_dict/working_data_dict_vae' + '.pkl', 'rb') as file_to_read:
    working_data_dict = pickle.load(file_to_read)


def load_transformed_data(data_folder):

    # Initialize an empty dictionary to store the data
    loaded_data_dict = {}

    # Loop through all files in the data folder
    for filename in os.listdir(data_folder):
        # Construct the full path of the file
        file_path = os.path.join(data_folder, filename)
        
        # Extract the subject ID and key from the filename
        # Assuming filenames are in the format 'sub_key.ext'
        sub, key = filename.split('_', 1)
        key = key.rsplit('.', 1)[0]  # Removes extension part to get the key
        
        #repeat
        key = key.rsplit('.', 1)[0]  # Removes extension part to get the key

        # Initialize sub dictionary if it doesn't exist
        if sub not in loaded_data_dict:
            loaded_data_dict[sub] = {}

        # Determine the file type and load accordingly
        if filename.endswith('.npy'):
            loaded_data_dict[sub][key] = np.load(file_path)
        elif filename.endswith('.pkl'):
            with open(file_path, 'rb') as file:
                loaded_data_dict[sub][key] = pickle.load(file)

        print(f"Loaded {key} from {file_path}")

    return loaded_data_dict


transform_masking=True
subject_ids = ["sub-001", "sub-002", "sub-003", "sub-004", "sub-005"]

mask_path = "mask_to_save/mask_him_005.nii.gz"
# mask_path = "mask_01.nii.gz"
if transform_masking:
    base_masker=working_data_dict["sub-001"]["masker"]
    selected_indices=base_masker.transform(nib.load(mask_path))
    for sub in subject_ids:
        working_data_dict[sub]["train_fmri_avg"]=working_data_dict[sub]["train_fmri"].squeeze()[:,selected_indices.squeeze().astype(np.uint8)]
        working_data_dict[sub]["test_fmri_avg"]=working_data_dict[sub]["test_fmri"].squeeze()[:,selected_indices.squeeze().astype(np.uint8)]
       
if transform_masking:
    # Ensure the data directory exists
    data_folder = 'data'
    os.makedirs(data_folder, exist_ok=True)

    # Assuming working_data_dict is your main dictionary and 'sub' is defined
    for sub in subject_ids:
        for key, value in working_data_dict[sub].items():
            # Convert the value to a numpy array if it's not already one 
            print(key)

            # Define the path for the output file
            file_path = os.path.join(data_folder, f'{sub}_{key}.npy')

            # Check if the value is a numpy array
            if isinstance(value, np.ndarray):
                # Save the numpy array to a file with .npy extension
                np.save(file_path + '.npy', value)
                print(f"Saved {key} as an array to {file_path}.npy")
            else:
                # Save other types of data using pickle with .pkl extension
                with open(file_path + '.pkl', 'wb') as file:
                    pickle.dump(value, file, protocol=pickle.HIGHEST_PROTOCOL)
                print(f"Saved {key} using pickle to {file_path}.pkl")

            print(f"Saved {key} to {file_path}")      

else:
    # Use the function
    data_folder = 'data'  # Specify the data folder path
    working_data_dict = load_transformed_data(data_folder)


df_captions = pd.read_csv('/srv/nfs-data/sisko/matteoc/music' + '/brain2music-captions.csv')
print(df_captions.head())
captions_array = df_captions.to_numpy()
caption_dict = {item[0].split('_')[0]: item[1] for item in captions_array}

for subj in subject_ids:

    train_stim_name_list = working_data_dict[subj]['train_stim_name']
    train_stim_caption_list = []

    for file_path in train_stim_name_list:
        genre_number = file_path.split('/')[-1].replace('.wav', '')
        caption = caption_dict.get(genre_number, "Caption not found")
        train_stim_caption_list.append(caption)

    working_data_dict[subj]['train_stim_caption'] = train_stim_caption_list


for subj in subject_ids:

    test_stim_name_list = working_data_dict[subj]['test_stim_name']
    test_stim_caption_list = []

    for file_path in test_stim_name_list:
        genre_number = file_path.split('/')[-1].replace('.wav', '')
        caption = caption_dict.get(genre_number, "Caption not found")
        test_stim_caption_list.append(caption)

    working_data_dict[subj]['test_stim_caption'] = test_stim_caption_list



def process_data(data_dict, key_suffix, features_key, genre_key, stim_name_key, fmri_key, stim_caption_key, feat_vae_key):
    # Create DataFrame for fMRI and stimulus names
    df = pd.DataFrame(data_dict[fmri_key], dtype=float)
    df['Stimulus'] = data_dict[stim_name_key]
    
    # Group by 'Caption' and calculate the mean for fMRI data
    fmri_avg = df.groupby('Stimulus').mean().reset_index()
    
    # Create DataFrame for audio features and stimulus names
    df_features = pd.DataFrame(data_dict[features_key], dtype=float)
    df_features['Stimulus'] = data_dict[stim_name_key]
    
    # Group by 'Caption' and calculate the mean for audio features
    features_avg = df_features.groupby('Stimulus').mean().reset_index()
    
    # Handle genres (assuming genre data is aligned with stimuli names)
    df_genre = pd.DataFrame(data_dict[genre_key], columns=['Genre'])
    df_genre['Stimulus'] = data_dict[stim_name_key]
    
    # Since genre should be consistent for the same caption, we can take the first occurrence
    genre_avg = df_genre.groupby('Stimulus').first().reset_index()

    # Handle genres (assuming genre data is aligned with stimuli names)
    df_caption = pd.DataFrame(data_dict[stim_caption_key], columns=['Caption'])
    df_caption['Stimulus'] = data_dict[stim_name_key]
    
    # Since genre should be consistent for the same caption, we can take the first occurrence
    caption_avg = df_caption.groupby('Stimulus').first().reset_index()

    df_vae = pd.DataFrame(data_dict[feat_vae_key].reshape(data_dict[feat_vae_key].shape[0],-1), dtype=float)
    df_vae['Stimulus'] = data_dict[stim_name_key]
    
    vae_avg = df_vae.groupby('Stimulus').mean().reset_index()
    
    return {
        key_suffix + '_audio_feat': features_avg.drop(columns='Stimulus').values,
        key_suffix + '_genre': genre_avg['Genre'].values,
        key_suffix + '_stim_name_avg': fmri_avg['Stimulus'].values,
        key_suffix + '_caption_avg': caption_avg['Caption'].values,
        key_suffix + '_fmri_avg': fmri_avg.drop(columns='Stimulus').values,
        key_suffix + '_vae_avg': vae_avg.drop(columns='Stimulus').values
    }


working_data_dict_avg = {}
for sub in subject_ids:
    working_data_dict_avg[sub] = {}

    # Process training data
    working_data_dict_avg[sub].update(
        process_data(
            working_data_dict[sub],
            'train',
            'train_audio_feat',
            'train_genre',
            'train_stim_name',
            'train_fmri_avg',
            'train_stim_caption',
            'train_audio_vae'
        )
    )

    # Process testing data
    working_data_dict_avg[sub].update(
        process_data(
            working_data_dict[sub],
            'test',
            'test_audio_feat',
            'test_genre',
            'test_stim_name',
            'test_fmri_avg',
            'test_stim_caption',
            'test_audio_vae'
        )
    )


# STRONGER FUNCTIONAL ALIGNMENT

target_sub="sub-001"

# X_train_aligned = [working_data_dict_avg[source_sub]["train_fmri_avg"]]
# X_test_aligned  = [working_data_dict_avg[source_sub]["test_fmri_avg"]]
X_train_aligned = []
X_test_aligned  = []


for source_sub in subject_ids:

    print(source_sub)
    source_train=working_data_dict_avg[source_sub]["train_fmri_avg"]
    target_train=working_data_dict_avg[target_sub]["train_fmri_avg"]

    source_test=working_data_dict_avg[source_sub]["test_fmri_avg"]
    target_test=working_data_dict_avg[target_sub]["test_fmri_avg"]

    
    aligner=RidgeCV(alphas=[1e2,1e3,1e4,5e4], fit_intercept=True)
    aligner.fit(source_train,target_train)

    aligned_source_test=aligner.predict(source_test)
    aligned_source_train=aligner.predict(source_train)

    aligned_source_train_adj = (aligned_source_train - aligned_source_train.mean(0)) / (1e-8 + aligned_source_train.std(0))
    aligned_source_train_adj = target_train.std(0) * aligned_source_train_adj + target_train.mean(0)

    # Align and adjust source_test dataset
    aligned_source_test_adj = (aligned_source_test - aligned_source_test.mean(0)) / (1e-8 + aligned_source_test.std(0))
    aligned_source_test_adj = target_train.std(0) * aligned_source_test_adj + target_train.mean(0)
    
    X_train_aligned.append(aligned_source_train_adj)
    X_test_aligned.append(aligned_source_test_adj)
    
#concatenate all

X_train_aligned = np.concatenate(X_train_aligned,0)
X_test_aligned = np.concatenate(X_test_aligned,0)

#concatenate all the other keys

train_audio_feat_aligned = np.concatenate([working_data_dict_avg[sub]["train_audio_feat"] for sub in subject_ids],0)
test_audio_feat_aligned = np.concatenate([working_data_dict_avg[sub]["test_audio_feat"] for sub in subject_ids],0)

train_genre_aligned = np.concatenate([working_data_dict_avg[sub]["train_genre"] for sub in subject_ids],0)
test_genre_aligned = np.concatenate([working_data_dict_avg[sub]["test_genre"] for sub in subject_ids],0)

train_stim_name_avg_aligned = np.concatenate([working_data_dict_avg[sub]["train_stim_name_avg"] for sub in subject_ids],0)
test_stim_name_avg_aligned = np.concatenate([working_data_dict_avg[sub]["test_stim_name_avg"] for sub in subject_ids],0)

train_caption_avg_aligned = np.concatenate([working_data_dict_avg[sub]["train_caption_avg"] for sub in subject_ids],0)
test_caption_avg_aligned = np.concatenate([working_data_dict_avg[sub]["test_caption_avg"] for sub in subject_ids],0)

train_vae_avg_aligned = np.concatenate([working_data_dict_avg[sub]["train_vae_avg"] for sub in subject_ids],0)
test_vae_avg_aligned = np.concatenate([working_data_dict_avg[sub]["test_vae_avg"] for sub in subject_ids],0)


audio_feat_caps = torch.load("/srv/nfs-data/sisko/matteoc/music/feature/audio_feat_clap_cap.pt")
caps_caption = np.load('/srv/nfs-data/sisko/matteoc/music/feature/text_caption_cap.npy').tolist()

print('audio_feat_caps.shape: ', audio_feat_caps.shape)

text_feat_cap_ldm = []
with torch.no_grad():
    for tx in tqdm.tqdm(caps_caption):
        text_ldm = musicldm_pipe._encode_prompt(tx, device=device, num_waveforms_per_prompt=1, do_classifier_free_guidance=False)
        text_feat_cap_ldm.append(text_ldm)

text_feat_cap_ldm=torch.stack(text_feat_cap_ldm).squeeze()


input_dim = 512
output_dim = 512
linear_layer = torch.nn.Linear(input_dim, output_dim).to(device)
criterion = torch.nn.MSELoss()
optimizer = optim.Adam(linear_layer.parameters(), lr=0.0001, weight_decay=1e-6)

# Training loop
num_epochs = 1000
for epoch in range(num_epochs):
    linear_layer.train()
    optimizer.zero_grad() 
    audio_feat_train_output = linear_layer(audio_feat_caps.to(device))
    loss = criterion(audio_feat_train_output, text_feat_cap_ldm.float()) 
    loss.backward()
    optimizer.step()
    
    if (epoch+1) % 20 == 0:
        print(f'Epoch [{epoch+1}/{num_epochs}], Loss: {loss.item()}')


train_text_ldm = []
with torch.no_grad():
    for tx in tqdm.tqdm(train_caption_avg_aligned):
        text_ldm = musicldm_pipe._encode_prompt(tx, device=device, num_waveforms_per_prompt=1, do_classifier_free_guidance=False)
        train_text_ldm.append(text_ldm)

test_text_ldm = []
with torch.no_grad():
    for tx in tqdm.tqdm(test_caption_avg_aligned):
        text_ldm = musicldm_pipe._encode_prompt(tx, device=device, num_waveforms_per_prompt=1, do_classifier_free_guidance=False)
        test_text_ldm.append(text_ldm)

train_text_feat_avg_aligned=torch.stack(train_text_ldm).squeeze()
test_text_feat_avg_aligned=torch.stack(test_text_ldm).squeeze()


linear_layer.eval()  
with torch.no_grad():  
    train_feature_finetuned = linear_layer(torch.tensor(train_audio_feat_aligned, dtype=torch.float32, device=device))
    test_loss = criterion(train_feature_finetuned, train_text_feat_avg_aligned)
    
print(f'Test Loss: {test_loss.item()}')

linear_layer.eval()  
with torch.no_grad():  
    test_feature_finetuned = linear_layer(torch.tensor(test_audio_feat_aligned, dtype=torch.float32, device=device))
    test_loss = criterion(test_feature_finetuned, test_text_feat_avg_aligned)
    
print(f'Test Loss: {test_loss.item()}')


X_train=X_train_aligned.copy()
X_test=X_test_aligned.copy()

brain_to_latent=Ridge(alpha=20)
brain_to_latent.fit(X_train, train_feature_finetuned.cpu())

audio_feat_to_generate = brain_to_latent.predict(X_test)
audio_feat_to_generate = torch.tensor(audio_feat_to_generate, dtype=torch.float32, device=device)


random.seed(42)
list_seed = [random.randint(0, 10000) for _ in range(100)]
track_list = np.arange(0, 60) 

model_path = "encoding_trained_model.joblib"
vm_loaded = load(model_path)
print("Modello caricato con successo.")

resampler = torchaudio.transforms.Resample(orig_freq=16000, new_freq=48000)
neg_prompt = "Low quality"    # Low quality, wadded, muffled and noisy audio
neg_prompt_embd = musicldm_pipe._encode_prompt(neg_prompt, device=device, num_waveforms_per_prompt=1, do_classifier_free_guidance=False)

duration = 30
# start_track = track_list[57]
for start_track in track_list:
    stop_track = start_track + 1
    corr_val = -1.0
    for seed in tqdm.tqdm(list_seed):
        file_name = os.path.splitext(os.path.basename(test_stim_name_avg_aligned[:60][start_track:stop_track][0]))[0]
        if seed == list_seed[0]:
            print('URL audio: ', test_stim_name_avg_aligned[:60][start_track:stop_track])
            print('Audio Caption: ', test_caption_avg_aligned[:60][start_track:stop_track])
            path_to_save_true = '/home/matteoc/genre-to-fmri/spectr_gen_bayes/spectr_true'
            audio_vera_wav = librosa.load(test_stim_name_avg_aligned[:60][start_track:stop_track][0], sr=16000)
            audio_widget_vera = Audio(audio_vera_wav[0][0:int(16000*(duration/2))], rate=16000)
            save_spect(audio_vera_wav[0][0:int(16000*(duration/2))], file_name, path_to_save_true, 'vera')
            display(audio_widget_vera)
        seed_gen = torch.Generator(device=device).manual_seed(int(seed))
        audio_pred_from_brain = musicldm_pipe(prompt_embeds=audio_feat_to_generate[start_track+0:stop_track+0],   # TODO: audio_feat_to_generate o test_text_feat_avg_aligned
                                    guidance_scale=2, # negative_prompt_embeds=neg_prompt_embd,
                                    generator=seed_gen, num_inference_steps=100, audio_length_in_s=duration/2).audios[0]
        audio_widget = Audio(audio_pred_from_brain, rate=16000)
        with torch.no_grad():
            audio_pred_from_brain_rs = audio_pred_from_brain[0:160000]
            audio_pred_from_brain_rs = resampler(torch.tensor(audio_pred_from_brain_rs))
            inputs = processor(audios=audio_pred_from_brain_rs.squeeze(), return_tensors="pt", sampling_rate=48_000)
            audio_features_bayes = model.get_audio_features(inputs.input_features.to(device)).cpu()  
        predictions = vm_loaded.predict(audio_features_bayes.numpy())
        predictions = torch.tensor(predictions)[:,selected_indices.squeeze().astype(np.uint8)]
        corr_coeff = np.corrcoef(predictions, X_test_aligned[:60][start_track:stop_track])[0,1]
        if corr_coeff >= corr_val:
            corr_val = corr_coeff
            path_to_save_pred = '/home/matteoc/genre-to-fmri/spectr_gen_bayes/human_metric/'
            print('Best correlation: ', corr_val)
            print('Best seed: ', seed)
            display(audio_widget)
            write(path_to_save_pred+file_name+'.wav', 16000, (audio_pred_from_brain * 32767).astype(np.int16))


