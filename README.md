# speech_emotion_recognition_system
<br>
Speech Emotion Recognition (SER) using Deep Learning is a desktop-based AI system that automatically analyzes human speech to detect the underlying emotional tone. This application bridges the fields of machine learning, signal processing, and human-computer interaction to create a tool that can "understand how someone feels" just by listening to a voice recording.

The goal of this project is to build an interactive, intelligent system capable of predicting emotions such as happy, sad, angry, fearful, and more — from raw audio inputs like .wav or .mp3 files. It leverages a pre-trained deep learning model built using Keras and trained on emotional speech datasets.

The application provides a Graphical User Interface (GUI) built using Tkinter, which makes it beginner-friendly and usable without any coding knowledge. Users can simply browse an audio file, let the model process it, and receive an emotion prediction instantly, along with real-time audio playback using SoundDevice.

What This Project Does?

Loads an audio file selected by the user
* Extracts key features using MFCCs (Mel-Frequency Cepstral Coefficients) — a proven method in speech analysis
* Feeds the features into a pre-trained neural network model
* Predicts the most probable emotion from a predefined list
* Plays back the audio clip to enhance the user experience

💡 Use Cases

* Affective computing research
* Human-robot interaction systems
* Call center emotion monitoring
* Mental health tools for emotion tracking
* Smart assistants and voice-based apps

🧠 How it Works Internally
Audio Input: The user selects an audio file using the GUI.

Preprocessing:
Audio is loaded and sampled using librosa.
MFCCs are extracted and shaped into a consistent format.

Model Inference:
A trained Keras .h5 model processes the MFCC features.
The model predicts an emotion class using softmax.

Output:
The predicted emotion is shown in the GUI.
The audio is played using sounddevice.

🧪 Trained Model Info
Input Shape: (40, 13) — representing 40 time frames × 13 MFCC coefficients
Output: One of the following 7 emotion classes:

😠 Angry

🤢 Disgust

😨 Fear

😄 Happy

😐 Neutral

😢 Sad

😲 Surprise

📌 Why This Project is Useful
Understanding emotions in voice can enhance digital systems in:
Customer experience analysis
Therapeutic applications
Security systems
Entertainment (games, interactive media)
It also serves as a strong base for students and researchers to build upon or integrate into larger emotion-aware systems.





