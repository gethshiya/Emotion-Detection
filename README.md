# 🎭 Emotion Detection from Speech and Video
# Overview
This project detects human emotions—such as happy, sad, angry, and neutral—from audio extracted from video files. It uses OpenAI Whisper for transcription, Librosa for audio feature extraction, and scikit-learn (SVM) for emotion classification.

# 🎯 Objectives
Transcribe speech from video/audio.

Extract audio features (MFCC, pitch).

Classify the speaker’s emotion.

Display the transcription, detected emotion, and spoken language.

# 🧠 Technologies Used
Technology	Purpose
Whisper	Transcribe speech and detect language
Librosa	Extract MFCC and pitch features
scikit-learn	Train & apply Support Vector Machine (SVM)
moviepy	Convert video files (MP4) to audio (WAV)
NumPy	Numerical operations and feature vectors

# 🚀 How It Works
Input: Video/audio file.

Convert: Extract audio from video using moviepy.

Transcribe: Use Whisper to generate transcription and detect language.

Extract Features: Get MFCC and pitch features with Librosa.

Predict Emotion: Use a trained SVM model to classify emotion.

Output: Print transcription, detected emotion, and language.

# 🛠 Installation
bash
Copy
Edit
pip install openai-whisper librosa scikit-learn moviepy numpy
# 📄 Sample Code
python
Copy
Edit
import whisper
import librosa
import numpy as np
from sklearn.svm import SVC
from sklearn.preprocessing import LabelEncoder
from moviepy.editor import AudioFileClip

# Load model
model = whisper.load_model("base")

# Functions
def transcribe_audio(audio_path):
    result = model.transcribe(audio_path)
    return result["text"], result["language"]

def extract_audio_features(audio_path):
    y, sr = librosa.load(audio_path, sr=16000)
    mfcc = np.mean(librosa.feature.mfcc(y=y, sr=sr, n_mfcc=13), axis=1)
    pitch, _ = librosa.piptrack(y=y, sr=sr)
    pitch = np.mean(pitch, axis=1)[:13]
    return np.concatenate([mfcc, pitch])

def train_emotion_classifier():
    X = np.random.rand(100, 26)
    y = np.random.choice(['happy', 'sad', 'angry', 'neutral'], size=100)
    le = LabelEncoder()
    y_encoded = le.fit_transform(y)
    clf = SVC(kernel='linear')
    clf.fit(X, y_encoded)
    return clf, le

def classify_emotion(features, clf, le):
    emotion_idx = clf.predict([features])[0]
    return le.inverse_transform([emotion_idx])[0]

def convert_mp4_to_wav(mp4_path, wav_path):
    AudioFileClip(mp4_path).write_audiofile(wav_path, codec='pcm_s16le')

def emotion_aware_speech_recognition(mp4_path):
    wav_path = "converted.wav"
    convert_mp4_to_wav(mp4_path, wav_path)
    transcription, language = transcribe_audio(wav_path)
    features = extract_audio_features(wav_path)
    emotion = classify_emotion(features, clf, le)
    print(f"Transcription: {transcription}")
    print(f"Language: {language}")
    print(f"Detected Emotion: {emotion}")

clf, le = train_emotion_classifier()
emotion_aware_speech_recognition("your_video.mp4")

# ⚠️ Limitations
Trained on simulated (fake) data, not real emotional recordings.

Detects only 4 basic emotions.

Not suitable for real-time/live input.

Ignores facial expressions and contextual text cues.

Sensitive to background noise.

# Input Link
 https://youtube.com/shorts/qe3lzbs47UY?si=N20f5ymqvEahgLrD

# 🔮 Future Enhancements
Train on real-world emotional datasets.

Detect more diverse and subtle emotions.

Add real-time processing via microphone.

Combine multimodal inputs: audio, text, video.

Create a user-friendly web interface or API.

Improve multilingual support and personalization.

# 📚 Conclusion
This project demonstrates a foundational approach to emotion detection using audio data. While limited by its training data and emotion scope, it sets the stage for more advanced, real-time, and multimodal emotion-aware systems.
