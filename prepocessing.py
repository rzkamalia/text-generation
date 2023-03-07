import tensorflow
from tensorflow.keras.preprocessing.text import Tokenizer
from tensorflow.keras.utils import pad_sequences

tokenizer = Tokenizer()

dataset = "data_text_generation.txt"
with open(dataset, "r") as file:
    text = file.read()  

text = text.lower().split('.')
tokenizer.fit_on_texts(text)

input_sequences = []
for seq in text:
    sequence = tokenizer.texts_to_sequences([seq])[0]
    for i in range(1, len(sequence)):
        n_gram_sequence = sequence[:i+1]
        input_sequences.append(n_gram_sequence)

padded_sequences = pad_sequences(input_sequences)