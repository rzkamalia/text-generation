from flask import Flask, render_template, request
import tensorflow
import numpy as np 

from prepocessing import *

model = tensorflow.keras.models.load_model('./models/best_text_generation_colab.h5')

def processAI(inputS):
    for _ in range(25):
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
    return inputS

app = Flask(__name__)

@app.route('/', methods = ['GET', 'POST'])
def index():
    if request.method == 'POST':
        sentence = request.form['sentence']
        return render_template('result.html', sentiment = processAI(sentence))
    return render_template('form.html')

app.run()