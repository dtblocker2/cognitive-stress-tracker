import pandas as pd
import numpy as np
import matplotlib.pylab as plt
import seaborn as sns

from glob import glob

import librosa
import librosa.display
import IPython.display as ipd

from itertools import cycle
    
sns.set_theme(style="white", palette=None)
color_pal = plt.rcParams["axes.prop_cycle"].by_key()["color"]
color_cycle = cycle(plt.rcParams["axes.prop_cycle"].by_key()["color"])

# Loading the audio file
audio_file = glob('test.wav')[0]  # assumes 'test.wav' is present in the directory

# Use librosa to load the audio
audio, sr = librosa.load(audio_file)

# Play audio in Jupyter/IPython
ipd.display(ipd.Audio(audio, rate=sr))

# Split silent and non-silent intervals
non_silent_intervals = librosa.effects.split(audio, top_db=20)
print(non_silent_intervals)
print(sr)

pause_duration=[]
for i in range(0,len(non_silent_intervals)-1):
    pause_duration.append(non_silent_intervals[i+1,0] - non_silent_intervals[i,1])

print(pause_duration)

avg_pause_duration = np.mean(pause_duration)/sr
print(avg_pause_duration)

#pitch info
pitches, magnitudes,voicing = librosa.pyin(audio, fmin=librosa.note_to_hz('C1'), fmax=librosa.note_to_hz('C8'))

print(pitches)
print(magnitudes)
print(voicing)

import speech_recognition as sr

# Initialize recognizer
recognizer = sr.Recognizer()

# Load your audio file
audio_file = "test.wav"

# Use AudioFile class
with sr.AudioFile(audio_file) as source:
    audio_data = recognizer.record(source)

# Perform Speech-to-Text using Google's API
try:
    text = recognizer.recognize_google(audio_data)
    print("Transcribed Text:\n", text)
except sr.UnknownValueError:
    print("Google Speech Recognition could not understand the audio.")
except sr.RequestError as e:
    print(f"Could not request results; {e}")

audio_file = 'test.wav'  # Replace with the actual file path
try:
    y, sr = librosa.load(audio_file)  # y: audio time series, sr: sampling rate
    audio_length = len(y) / sr  # Calculate length in seconds
    print(f"Length of the audio file: {audio_length:.2f} seconds")
except FileNotFoundError:
    print("Error: The specified audio file could not be found.")

word_count = len(text.split())

speech_rate = word_count/audio_length
#speech rate
print(f"Speech Rate is {speech_rate} wps")

pitches, magnitudes = librosa.piptrack(y=y, sr=sr)
pitch_values = pitches[pitches > 0]
pitch_variability = np.std(pitch_values) if len(pitch_values) > 0 else 0

print(pitch_variability)

def detect_placeholder_words(text):
    placeholder_words = ['thing', 'stuff', 'whats', 'you know', 'um', 'uh', 'like', 'uhh', 'that thing', 'this thing']
    
    # Tokenize and check for placeholders
    detected =[]
    for i in placeholder_words:
        if i in text:
            detected.append(i)
    
    return detected

# Example usage
transcription = "I was thinking about that thing, you know, the one we talked about earlier."
placeholders = detect_placeholder_words(transcription)
print(f"Detected Placeholder Words: {placeholders}")

filler_count = len(placeholders)
word_count = len(transcription)

features=[]
features.append({
        "Audio File": audio_file,
        "Speech Rate": speech_rate,
        "Avg Pause Duration": avg_pause_duration,
        "Pitch Variability": pitch_variability,
        "Filler Word Count": filler_count,
        "Word Count": word_count
    })

# Convert to DataFrame
feature_matrix = pd.DataFrame(features)
print(feature_matrix)

from sklearn.preprocessing import StandardScaler
from sklearn.cluster import KMeans
import pandas as pd

# Example feature matrix (replace with your real data)
feature_matrix = pd.DataFrame({
    'speech_rate': [2.1, 1.9, 1.1, 2.5, 1.0],
    'avg_pause_duration': [0.34, 0.51, 0.8, 0.25, 0.9],
    'pitch_variability': [25, 20, 8, 30, 5],
    'filler_word_count': [3, 2, 6, 1, 8]
})

# Scale features for better clustering
scaler = StandardScaler()
scaled_features = scaler.fit_transform(feature_matrix)

# K-Means clustering
kmeans = KMeans(n_clusters=2, random_state=42)
feature_matrix['Cluster'] = kmeans.fit_predict(scaled_features)

print(feature_matrix)

import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

# Example feature matrix
feature_matrix = pd.DataFrame({
    'speech_rate': [2.1, 1.9, 1.1, 2.5, 1.0],
    'avg_pause_duration': [0.34, 0.51, 0.8, 0.25, 0.9],
    'pitch_variability': [25, 20, 8, 30, 5],
    'filler_word_count': [3, 2, 6, 1, 8],
    'Cluster': [0, 0, 1, 0, 1],
    'Anomaly': [1, 1, -1, 1, -1]
})

# Scatter plot
plt.figure(figsize=(8, 5))
sns.scatterplot(
    data=feature_matrix,
    x='avg_pause_duration',
    y='speech_rate',
    hue='Cluster',
    style='Anomaly',
    palette='deep',
    s=100
)
plt.title('Pause Duration vs Speech Rate')
plt.xlabel('Average Pause Duration (sec)')
plt.ylabel('Speech Rate (words/sec)')
plt.legend(title='Cluster / Anomaly')
plt.grid(True)
plt.show()

from fastapi import FastAPI, UploadFile, File
import librosa
import numpy as np
from sklearn.ensemble import IsolationForest

app = FastAPI()

# Dummy pre-trained Isolation Forest (normally you'd load a real one!)
iso_forest = IsolationForest(contamination=0.2, random_state=42)

# Dummy fit with placeholder data
dummy_data = np.array([
    [2.1, 0.3, 25, 3], 
    [1.9, 0.5, 20, 2], 
    [1.0, 0.9, 5, 8]
])
iso_forest.fit(dummy_data)

@app.post("/predict_risk/")
async def predict_cognitive_risk(audio_file: UploadFile = File(...)):
    # Load audio
    contents = await audio_file.read()
    with open("temp_audio.wav", "wb") as f:
        f.write(contents)

    y, sr = librosa.load("temp_audio.wav")

    # Basic feature extraction
    duration = librosa.get_duration(y=y, sr=sr)
    intervals = librosa.effects.split(y, top_db=25)
    pause_durations = []
    for i in range(len(intervals) - 1):
        pause = (intervals[i+1][0] - intervals[i][1]) / sr
        pause_durations.append(pause)
    avg_pause = np.mean(pause_durations) if pause_durations else 0

    pitches, magnitudes = librosa.piptrack(y=y, sr=sr)
    pitch_values = pitches[pitches > 0]
    pitch_variability = np.std(pitch_values) if len(pitch_values) > 0 else 0

    # Dummy text info for word count and fillers (simulate)
    word_count = 20  # assume a speech-to-text output length
    filler_word_count = 4  # assume you parsed and counted fillers

    # Assemble feature vector
    features = np.array([[word_count/duration, avg_pause, pitch_variability, filler_word_count]])

    # Predict risk
    prediction = iso_forest.predict(features)  # -1 = high risk, 1 = low risk

    risk = "High Risk of Cognitive Stress/Decline" if prediction[0] == -1 else "Low Risk (Normal)"

    return {"risk_assessment": risk}

def normalize(value, low, high):
    return max(0, min(1, (value - low) / (high - low)))

speech_ratef = 1-normalize(speech_rate,1.5,3.0)
avg_pause_durationf = normalize(avg_pause_duration,0.2,0.8)
filler_countf = normalize(filler_count,0,5)
word_countf = normalize(word_count,20,200)
pitch_variabilityf = 1-normalize(pitch_variability,5,35)

risk_factor = (speech_ratef*0.2+avg_pause_durationf*0.2+filler_countf*0.2+word_countf*0.2+pitch_variabilityf*0.2)*100
print(f"Risk factor: {risk_factor.round(2)}%")