def detect_substitutions(text):
    # Common placeholders or substitution signs
    substitution_clues = ["thing", "stuff", "what's it called", "you know", "whatchamacallit", 
                          "that guy", "that place", "uh", "um", "err"]
    
    found = []
    for clue in substitution_clues:
        if clue in text.lower():
            found.append(clue)
    
    return found

# Example usage
transcription = "I went to that place — you know, the one with the stuff — but I forgot the name."
substitutions = detect_substitutions(transcription)

print("Possible Word Substitution Signs:", substitutions)
