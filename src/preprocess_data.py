import re
import pandas as pd
from ocr_extraction import extract_texts

def preprocess_texts(texts):
    """Basic preprocessing for texts."""
    processed_texts = []
    
    for text in texts:
        # Remove any special characters, keeping English and Malayalam characters
        text = re.sub(r'[^\w\s]', '', text)  # This will keep all word characters, including non-English characters

        # Normalize white spaces and convert to lowercase (only for English characters)
        text = text.strip()

        # Tokenization (split into words)
        tokens = text.split()

        # Rejoin tokens into cleaned sentence
        processed_text = ' '.join(tokens)
        
        processed_texts.append(processed_text)
    
    return processed_texts

if __name__ == "__main__":
    # Load the CSV file containing the image information (replace with your actual CSV path)
    data = pd.read_csv('../data/data.csv')

    # Extract texts from images using OCR
    texts = extract_texts(data)

    # Preprocess the extracted texts
    processed_texts = preprocess_texts(texts)
    
    # Display the results
    for original, processed in zip(texts, processed_texts):
        print(f'Original: {original}\nProcessed: {processed}\n')
