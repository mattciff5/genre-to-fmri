import streamlit as st
import numpy as np
import os
import random
import wave
from io import BytesIO
import json

# Function to load audio files
# def load_audio(file_path):
#     with open(file_path, 'rb') as f:
#         audio_bytes = f.read()
#     return audio_bytes

def load_audio(file_path, duration=10):
    with wave.open(file_path, 'rb') as wav_file:
        sample_rate = wav_file.getframerate()
        n_channels = wav_file.getnchannels()
        sample_width = wav_file.getsampwidth()

        frames_to_read = sample_rate * duration
        audio_frames = wav_file.readframes(frames_to_read)
        audio_samples = np.frombuffer(audio_frames, dtype=np.int16)
        
        if n_channels == 2:
            audio_samples = audio_samples.reshape(-1, 2)
        
        output = BytesIO()
        with wave.open(output, 'wb') as out_wav:
            out_wav.setnchannels(n_channels)
            out_wav.setsampwidth(sample_width)
            out_wav.setframerate(sample_rate)
            out_wav.writeframes(audio_samples.tobytes())
        
        return output.getvalue()

# Set the title of the app
st.title("Audio Similarity Comparison")

# Load the numpy array once
test_stim_name_avg_aligned = np.load('/srv/nfs-data/sisko/matteoc/music/test_stim_name_avg_aligned.npy', allow_pickle=True)

# Generate path
generate_path = "/home/matteoc/genre-to-fmri/spectr_generative/human_metric/"

# Initialize session state variables
if 'name' not in st.session_state:
    name = st.text_input("Enter your name")
    st.session_state.name = name 

if 'track' not in st.session_state:
    st.session_state.track = 0  # Start with the first track

if 'selected_choices' not in st.session_state:
    st.session_state.selected_choices = []

if 'submitted' not in st.session_state:
    st.session_state.submitted = False

if 'selected_choice' not in st.session_state:
    st.session_state.selected_choice = None

if 'rand_track' not in st.session_state:
    st.session_state.rand_track = None

if 'correct_ones' not in st.session_state:
    st.session_state.correct_ones = 0

track_number = 60

if st.session_state.track < track_number:
    indexes = [i for i in range(track_number) if i != st.session_state.track]
    if st.session_state.rand_track is None:
        st.session_state.rand_track = random.choice(indexes)
        print('rand_track: ', st.session_state.rand_track)

    stimulus = test_stim_name_avg_aligned[:track_number][st.session_state.track:st.session_state.track+1]
    random_gen = test_stim_name_avg_aligned[:track_number][st.session_state.rand_track:st.session_state.rand_track+1]
    file_name_stim = os.path.splitext(os.path.basename(stimulus[0]))[0]
    rand_name_stim = os.path.splitext(os.path.basename(random_gen[0]))[0]

    audio1_path = stimulus[0]
    audio2_path = generate_path + file_name_stim + '.wav'
    audio3_path = generate_path + rand_name_stim + '.wav'

    # Load the audio files
    audio1 = load_audio(audio1_path)
    audio2 = load_audio(audio2_path)
    audio3 = load_audio(audio3_path)

    if audio1 and audio2 and audio3:
        st.header("Listen to the Audios")

        # Display the audio player for the first audio
        st.subheader("Reference Audio")
        st.audio(audio1, format='audio/mp3')  # Ensure audio1 is defined

        # Display the audio players for the comparison audios
        st.subheader("Comparison Audio 1")
        st.audio(audio2, format='audio/mp3')  # Ensure audio2 is defined

        st.subheader("Comparison Audio 2")
        st.audio(audio3, format='audio/mp3')  # Ensure audio3 is defined

        if not st.session_state.submitted:
            # Ask the user to select which audio is similar to the reference audio
            st.header("Which audio is similar to the reference audio?")
            options = ['Generative Audio 1', 'Generative Audio 2']
            index = 0
            if st.session_state.selected_choice is not None:
                index = options.index(st.session_state.selected_choice)

            choice = st.radio(
                "Choose one:",
                options,
                index=index,
                key='choice'
            )

            if st.button('Submit'):
                st.session_state.selected_choice = choice
                st.session_state.selected_choices.append(choice)
                st.session_state.submitted = True

        if st.session_state.submitted:
            st.write(f'You selected: {st.session_state.selected_choice}')
            st.write('Selection has been recorded.')
            if st.session_state.selected_choice == 'Generative Audio 1':
                 st.session_state.correct_ones += 1
            st.session_state.track += 1  # Move to the next track
            st.session_state.submitted = False  # Reset the submission state
            st.session_state.selected_choice = None  # Reset the selected choice for the next track
            st.session_state.rand_track = None
            st.rerun()  # Reload the app with the next track

    else:
        st.write("Failed to fetch all audio files.")
else:
    st.write("All tracks have been processed.")
    st.write("Selections:", st.session_state.selected_choices)

    # Save the data as a JSON file
    if st.session_state.name:
        data = {
            'name': st.session_state.name,
            'selected_choices': st.session_state.selected_choices,
            'correct_ones': st.session_state.correct_ones
        }
        
        with open(f"/srv/nfs-data/sisko/matteoc/music/{st.session_state.name}_human_metric.json", "w") as f:
            json.dump(data, f)
        
        st.write("Results have been saved.")

# streamlit run human_metric.py
