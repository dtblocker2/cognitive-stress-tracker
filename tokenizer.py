def find_repeated_words(text):
    # Convert text to lowercase and split it into words
    words = text.lower().split()
    
    # Create a dictionary to store word counts
    word_count = {}
    
    # Iterate over each word in the text
    for word in words:
        # If the word is already in the dictionary, increase its count
        if word in word_count:
            word_count[word] += 1
        else:
            word_count[word] = 1
    
    # Filter words that appear more than once
    repeated_words = {word: count for word, count in word_count.items() if count > 1}
    
    return repeated_words

# Example usage
speech = "banglore hello hello"
repeats = find_repeated_words(speech)
print("Repeated Words:", repeats)
