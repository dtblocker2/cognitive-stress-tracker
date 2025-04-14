import pandas as pd
import numpy as np
from glob import glob
import librosa
import speech_recognition as sr2

#load audio file
audio_file2='old_man_test.wav'
audio_file = glob(audio_file2)[0]
audio, sr = librosa.load(audio_file)

# Split silent and non-silent intervals
non_silent_intervals = librosa.effects.split(audio, top_db=20)

# average pause duration
pause_duration=[]
for i in range(0,len(non_silent_intervals)-1):
    pause_duration.append(non_silent_intervals[i+1,0] - non_silent_intervals[i,1])
avg_pause_duration = np.mean(pause_duration)/sr

#pitch info
pitches, magnitudes,voicing = librosa.pyin(audio, fmin=librosa.note_to_hz('C1'), fmax=librosa.note_to_hz('C8'))

# transcription using google
recognizer = sr2.Recognizer()
with sr2.AudioFile(audio_file2) as source:
    audio_data = recognizer.record(source)
try:
    transcription = recognizer.recognize_google(audio_data)
    print("Transcribed Text:\n", transcription)
except sr2.UnknownValueError:
    print("Google Speech Recognition could not understand the audio.")
except sr2.RequestError as e:
    print(f"Could not request results; {e}")

# audio length
audio_length = (non_silent_intervals[-1,-1] - non_silent_intervals[0,0] )/ sr
print(f"Length of the audio file: {audio_length:.2f} seconds")
word_set = transcription.split()
word_count = len(word_set)
unique_words = len(set(word_set))
vocabulary = (unique_words/word_count)
speech_rate = word_count/audio_length

#pitch variability
pitches, magnitudes = librosa.piptrack(y=audio, sr=sr)
pitch_values = pitches[pitches > 0]
pitch_variability = np.std(pitch_values) if len(pitch_values) > 0 else 0

#filler words detector
def detect_placeholder_words(text):
    placeholder_words = ['thing', 'stuff', 'whats', 'you know', 'um', 'uh', 'like', 'uhh', 'that thing', 'this thing','so']
    detected =[]
    for i in placeholder_words:
        if i in text:
            detected.append(i)
    return detected
placeholders = detect_placeholder_words(transcription.lower())
filler_count = len(placeholders)
word_count = len(set(transcription.split()))

# Convert to DataFrame
features={
        "Audio File": [],
        "Speech Rate": [],
        "Avg Pause Duration": [],
        "Pitch Variability": [],
        "Filler Word Count": [],
        "Vocabulary": [],
    }
features["Audio File"].append(audio_file)
features["Speech Rate"].append(speech_rate)
features["Avg Pause Duration"].append(avg_pause_duration)
features["Pitch Variability"].append(pitch_variability)
features["Filler Word Count"].append(filler_count)
features["Vocabulary"].append(vocabulary)
feature_matrix = pd.DataFrame(features)

#normalization
def normalize(value, low, high):
    return max(0, min(1, (value - low) / (high - low)))
speech_ratef = 1-normalize(speech_rate,1.5,3.0)
avg_pause_durationf = normalize(avg_pause_duration,0.2,0.8)
filler_countf = normalize(filler_count,1,5)
vocabularyf = 1-normalize(vocabulary,0.53,0.87)
pitch_variabilityf = 1-normalize(pitch_variability,5,35)

#risk factor
risk_factor = (speech_ratef*0.25+avg_pause_durationf*0.25+filler_countf*0.2+vocabularyf*0.15+pitch_variabilityf*0.15)*100
print(f"Risk factor: {round(risk_factor,2)}%")