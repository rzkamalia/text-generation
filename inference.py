import numpy as np
import tensorflow

from prepocessing import *

model = tensorflow.keras.models.load_model('./models/best_text_generation_colab.h5')

inputS = "Super Junior is"
next_words = 25

for _ in range(next_words):
    sequence = tokenizer.texts_to_sequences([inputS])[0]
    padded_sequences = pad_sequences([sequence])
    pred = model.predict(padded_sequences)
    pred = np.argmax(pred, axis = 1)
    output_word = ""
    for word, index in tokenizer.word_index.items():
        if index == pred:
            output_word = word
            break
    inputS += " " + output_word

print(inputS)