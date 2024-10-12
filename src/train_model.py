import numpy as np
from keras.models import Sequential
from keras.layers import Dense, LSTM, Embedding, SpatialDropout1D
import pandas as pd
from preprocess_data import preprocess_texts
from ocr_extraction import extract_texts

# Load your CSV data
file_path = '../data/data.csv'  # Update this path
data = pd.read_csv(file_path)

# Extract texts using OCR
texts = extract_texts(data)

# Preprocess the extracted texts
processed_texts = preprocess_texts(texts)

# Assuming labels are in a column in the CSV, like 'label' (Update as necessary)
labels = data['label'].values  # You need to have these labels defined in your data

# Prepare your input data for the model (you may need to adjust this based on text length and complexity)
# Convert texts to sequences (e.g., using a Tokenizer from Keras)
from keras.preprocessing.text import Tokenizer
from keras.preprocessing.sequence import pad_sequences

tokenizer = Tokenizer(num_words=1000, oov_token="<OOV>")
tokenizer.fit_on_texts(processed_texts)
sequences = tokenizer.texts_to_sequences(processed_texts)
X = pad_sequences(sequences, maxlen=100)  # Set max length for input data

# Define and compile your model (you can adjust this based on the complexity of your task)
model = Sequential()
model.add(Embedding(input_dim=1000, output_dim=64))
model.add(SpatialDropout1D(0.2))
model.add(LSTM(100, dropout=0.2, recurrent_dropout=0.2))
model.add(Dense(1, activation='sigmoid'))  # Adjust for binary classification

model.compile(loss='binary_crossentropy', optimizer='adam', metrics=['accuracy'])

# Train your model
model.fit(X, labels, epochs=5, batch_size=32)

# Save the trained model
model.save('../models/text_detection_model.h5')
