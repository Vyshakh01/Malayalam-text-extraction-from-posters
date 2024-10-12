import cv2
import pytesseract
import os

# Set Tesseract path if needed (adjust path based on your environment)
pytesseract.pytesseract.tesseract_cmd = r'C:\ Program Files\ Tesseract-OCR\ tesseract.exe'

def ocr_image(image_path):
    """Extract text from a given image using Tesseract OCR."""
    if os.path.exists(image_path):
        image = cv2.imread(image_path)
        if image is None:
            print(f"Error: Unable to load image {image_path}. Skipping...")
            return ""

        # Convert image to grayscale
        gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

        # Apply thresholding to make the text clearer for Tesseract
        gray = cv2.threshold(gray, 0, 255, cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU)[1]

        # Extract text from the processed grayscale image
        text = pytesseract.image_to_string(gray, lang='mal')
        
        if text.strip() == "":
            print(f"No text extracted from {image_path}.")
        else:
            print(f"Extracted text from {image_path}: {text}")

        return text
    else:
        print(f"Image does not exist: {image_path}")
        return ""

def extract_texts(data):
    """Extract texts from all images listed in the data."""
    texts = []
    
    for index, row in data.iterrows():
        image_path = f'../data/images/{row["id"]}.png'
        
        # Check if the image file exists
        if os.path.exists(image_path):
            text = ocr_image(image_path)  # Extract text if the image exists
            texts.append(text)
        else:
            print(f'Image {image_path} not found, skipping...')
            texts.append("")  # Append an empty string if the image doesn't exist

    return texts
