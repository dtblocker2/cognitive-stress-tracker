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
