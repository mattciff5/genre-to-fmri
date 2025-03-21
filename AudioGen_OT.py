import wandb
import torch.optim as optim
import geomloss  # Library per differentiable OT in PyTorch
import torch.nn as nn
import torch
import numpy as np
import random
from sklearn.linear_model import Ridge
from sklearn.neighbors import NearestNeighbors
import torch.nn.functional as F
import numpy as np
import scipy
import scipy.io as sio
import matplotlib.pyplot as plt
import nibabel as nib

import os
from os.path import join as opj
from os.path import join, exists, split
import time
import urllib.request
import warnings
from pprint import pprint
import zipfile
import glob
warnings.filterwarnings('ignore')
from transformers import AutoFeatureExtractor, ClapModel
import torch
import torchaudio

from sklearn.linear_model import LogisticRegression, RidgeCV, Ridge
from sklearn.metrics import classification_report
from sklearn.preprocessing import LabelEncoder


import pandas as pd
from nilearn import maskers
from nilearn import plotting
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


# -----------------------------------------------------------------------------------------
transform_masking=True
subject_ids = ["sub-001", "sub-002", "sub-003", "sub-004", "sub-005"]

# mask_path = "mask_to_save/mask_top_512_voxels.nii.gz"
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


# -----------------------------------------------------------------------------------------
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


# -----------------------------------------------------------------------------------------

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




X_train=X_train_aligned.copy()
X_test=X_test_aligned.copy()

device = torch.device("cuda:3" if torch.cuda.is_available() else "cpu")
torch.manual_seed(42)

X_train_torch = torch.tensor(X_train, dtype=torch.float32).to(device)
X_test_torch = torch.tensor(X_test, dtype=torch.float32).to(device)
train_audio_feat_torch = torch.tensor(train_audio_feat_aligned, dtype=torch.float32).to(device)

# Modello Ridge per mappare brain activity → latent space
brain_to_latent = Ridge(alpha=20)
brain_to_latent.fit(X_train, train_audio_feat_aligned)
audio_feat_pred_train = brain_to_latent.predict(X_train)
audio_feat_pred_train_torch = torch.tensor(audio_feat_pred_train, dtype=torch.float32).to(device)
audio_feat_pred_test = brain_to_latent.predict(X_test)
audio_feat_pred_test_torch = torch.tensor(audio_feat_pred_test, dtype=torch.float32).to(device)


# Matrice di similarità con cosine similarity
def cosine_similarity_matrix(A, B):
    A_norm = F.normalize(A, dim=1)
    B_norm = F.normalize(B, dim=1)
    return torch.mm(A_norm, B_norm.T)


# NT-Xent Loss (Contrastive)
def contrastive_loss(S, tau):
    S_exp = torch.exp(S / tau)
    loss = -torch.log(torch.diag(S_exp) / S_exp.sum(dim=1))
    return loss.mean()



# Imposta il seed per la riproducibilità
def set_seed(seed=42):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)  

set_seed(42)

# Configurazione Sweep di wandb
sweep_config = {
    "method": "grid",  # Usa "random" per una ricerca casuale
    "metric": {"name": "Total Loss", "goal": "minimize"},
    "parameters": {
        "blur": {"values": [0.5]},
        "lr": {"values": [1e-4, 1e-3]},
        "tau": {"values": [0.01, 0.05, 0.07, 0.1, 0.2, 0.3, 0.5]},
        "dropout": {"values": [0.3, 0.5, 0.7]}
    }
}

# Definizione del modello
class ContrastiveOT(nn.Module):
    def __init__(self, input_dim, output_dim, dropout):
        super().__init__()
        self.output = nn.Linear(input_dim, output_dim, bias=True)
        self.dropout = nn.Dropout(p=dropout)  
    
    def forward(self, x):       
        x = self.dropout(x)    
        return self.output(x)  

wandb.login()
sweep_id = wandb.sweep(sweep_config, project='music-genre-ot')

def train():
    wandb.init()
    config = wandb.config
    
    # Definizione della loss Sinkhorn con il parametro blur
    sinkhorn = geomloss.SamplesLoss(loss="sinkhorn", p=2, blur=config.blur)
    
    # Inizializzazione del modello con dropout variabile
    brain_to_latent_torch = ContrastiveOT(
        input_dim=X_train_torch.shape[1], 
        output_dim=train_audio_feat_aligned.shape[1], 
        dropout=config.dropout
    ).to(device)
    optimizer = optim.Adam(brain_to_latent_torch.parameters(), lr=config.lr)
    
    # Training
    n_epochs = 150
    for epoch in range(n_epochs):
        optimizer.zero_grad()
        audio_feat_pred_train_torch = brain_to_latent_torch(X_train_torch)
        ot_loss = sinkhorn(audio_feat_pred_train_torch, train_audio_feat_torch)
        S = cosine_similarity_matrix(audio_feat_pred_train_torch, train_audio_feat_torch)
        contrastive_loss_value = contrastive_loss(S, config.tau)
        loss = ot_loss + contrastive_loss_value
        loss.backward()
        optimizer.step()
        
        # Logging su wandb
        wandb.log({"Epoch": epoch+1, "Total Loss": loss.item(), 
                    "OT Loss": ot_loss.item(), "Contrastive Loss": contrastive_loss_value.item()})
    
    # Calcolo della distanza prima e dopo OT
    with torch.no_grad():
        audio_feat_pred_ot = brain_to_latent_torch(X_test_torch)
        audio_feat_pred_ot = audio_feat_pred_ot.cpu()
        brain_to_latent = Ridge(alpha=20)
        brain_to_latent.fit(X_train, train_audio_feat_aligned)
        audio_feat_pred = brain_to_latent.predict(X_test)
        
        nbrs = NearestNeighbors(n_neighbors=5, metric='cosine').fit(test_audio_feat_aligned[:60])
        distances_before, _ = nbrs.kneighbors(audio_feat_pred)
        distances, top_indices = nbrs.kneighbors(audio_feat_pred_ot)
        
        mean_distance_before = distances_before.mean()
        mean_distance_after = distances.mean()

        top_indices_3 = top_indices[:,:3]
        top_indices_3subj = top_indices_3.reshape([5, 60, 3])
        
        # Ciclo su tutti i soggetti per calcolare la media dei successi top-1 e top-3
        count_top_1_total = 0
        count_top_3_total = 0
        for sub_idx in range(5):
            count_top_1 = 0
            count_top_3 = 0
            for idx in range(60):
                file_name = test_stim_name_avg_aligned[:60][idx].split('/')[-1].replace('.wav', '')
                file_names_retrieved = [
                    file_path.split('/')[-1].replace('.wav', '') for file_path in test_stim_name_avg_aligned[:60][top_indices_3subj[sub_idx][idx]]
                ]
                if file_name == file_names_retrieved[0]:
                    count_top_1 += 1
                if file_name in file_names_retrieved:
                    count_top_3 += 1
            count_top_1_total += count_top_1
            count_top_3_total += count_top_3
        
        mean_top_1 = count_top_1_total / 5
        mean_top_3 = count_top_3_total / 5
        
        wandb.log({
            "Mean Distance Before OT": mean_distance_before,
            "Mean Distance After OT": mean_distance_after,
            "Mean Top-1 Accuracy": mean_top_1,
            "Mean Top-3 Accuracy": mean_top_3
        })
            
def main():
    wandb.agent(sweep_id, function=train, count=None) 

if __name__ == "__main__":
    main()
