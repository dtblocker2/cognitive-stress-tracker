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
