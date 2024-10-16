# genre-to-fmri

The "multisubject_align.ipynb" sheet code contains all the project pipeline, resumed in the following figure:

![pipeline](https://github.com/user-attachments/assets/0f6afd02-e933-4e45-8630-8b81cc8335b3)

In particular, we describe the main steps:

- Loading and preprocessing the fMRI and audio data, calling the "load_data" function from the "data_agg.py" code. For each subject, fMRI data is loaded, masked using NiftiMasker, and saved in a structured format. A matrix is constructed based on stimulus onsets, aligning fMRI data with specific genre-related events, ensuring temporal correspondence. The fMRI data undergoes standardization and detrending. Corresponding audio data is loaded for each event and passed through a pre-trained ClapModel (laion/larger_clap_music_and_speech) for feature extraction. Resampling is applied to align audio sample rates before feature extraction. The data is separated into training and test sets for both fMRI and audio features, along with genre labels.

- Using a base masker from one subject, the code undoes and reapplies the masking process for both training and testing fMRI data. This step ensures that all fMRI data is transformed to the same number of voxels for consistency.

- A specific mask (from the "mask_01.nii.gz" file computed in the "CLAP.ipynb" sheet code) is loaded, and fMRI data is reduced to selected voxels as defined by this mask. The masked fMRI data is then averaged across time points for each subject. The mask come out from the encoding model using Ridge regression and selecting voxels based on R² scores. The "CLAP.ipynb" code imports the RidgeCV regression model from the Himalaya package. The R² scores are transformed back into brain space using NiftiMasker and Nilearn, and threshold is applied to the smoothed R² image. 

- The function "process_data" organizes the data into structured formats, where fMRI data, audio features, and genre labels are grouped and averaged by stimulus (music events). This averaging helps in creating consistent input-output pairs for further modeling.

- Functional Alignment to align the brain activity (fMRI data) of different subjects to a common reference subject, sub-001. A loop iterates through all the subjects, aligning their fMRI data to the target one. A RidgeCV model is initialized (alphas=[1e2,1e3,1e4,5e4]) with a range of alphas (regularization parameters) and is used to align the source subject’s brain activity to the target subject.

- A Ridge regression model is initialized with an alpha value of 20, and it is trained to map the aligned fMRI data to the audio features in the decoding approach. After training, the model is used to predict audio features from the test fMRI data. 

- The k-nearest neighbors algorithm (NearestNeighbors) is applied to the audio features to identify the closest matches between the predicted audio features and the test set audio features. The function "calculate_retr_accuracy" computes the accuracy of genre retrieval by checking if the predicted genre matches the true genre. The accuracy is calculated for both Top-1 (the most likely prediction) and Top-3 (the top 3 closest predictions).










