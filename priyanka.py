import os
import tkinter as tk
from tkinter import filedialog
import numpy as np
import librosa
from keras.models import load_model
import sounddevice as sd

# Load the trained model
model = load_model('emotion_model_keras_format')

# Define emotions
emotions = ['angry', 'disgust', 'fear', 'happy', 'neutral', 'sad', 'surprise']

# Function to extract MFCC features from audio file
def extract_features(file_path):
    audio, sr = librosa.load(file_path, sr=None)
    mfccs = librosa.feature.mfcc(y=audio, sr=sr, n_mfcc=13)
    pad_width = 40 - mfccs.shape[1]
    if pad_width > 0:
        mfccs = np.pad(mfccs, pad_width=((0, 0), (0, pad_width)))
    else:
        mfccs = mfccs[:, :40]
    return mfccs

# Function to classify emotion
def classify_emotion(file_path):
    features = extract_features(file_path)
    features = np.expand_dims(features, axis=0)  # Add batch dimension
    predicted_class = np.argmax(model.predict(features), axis=1)[0]
    emotion = emotions[predicted_class]
    return emotion

# Function to handle button click event
def browse_file():
    file_path = filedialog.askopenfilename(filetypes=[("Audio Files", "*.wav;*.mp3")])
    if file_path:
        emotion = classify_emotion(file_path)
        result_label.config(text=f'Predicted Emotion: {emotion}')

        # Play the audio using sounddevice
        audio, _ = librosa.load(file_path, sr=None)
        sd.play(audio, samplerate=_)
        sd.wait()

# Create the main window
root = tk.Tk()
root.title("Speech Emotion Recognition")
root.geometry("400x300")
root.config(bg="#BB78FF")


# Create and configure the GUI components
browse_button = tk.Button(root, text="Browse",font=("ARIAL", 12,"bold"), command=browse_file,bg="#D2FFFF")
result_label = tk.Label(root, text="Predicted Emotion: ",font=("ARIAL", 15,"bold"),bg="#BB78FF")

# Place the components in the window
browse_button.pack(pady=20)
result_label.pack(pady=20)

# Run the GUI application
root.mainloop()
