from tensorflow.keras.layers import Embedding, LSTM, Dense, Bidirectional
from tensorflow.keras.models import Sequential
from tensorflow.keras.losses import CategoricalCrossentropy
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.callbacks import ModelCheckpoint
from tensorflow.keras.utils import to_categorical

import os

from prepocessing import *

# Embedding: input_dim = number of vocab that used in dataset / output_dim = embedding vector size
# LSTM: recurrent_activation = 'sigmoid' untuk di forget, input, dan output gate

if not os.path.exists('./models/'):
    os.makedirs('./models/')

def train(padded_sequences):
    inputS = padded_sequences[:,:-1]
    labelS = padded_sequences[:, -1]
    labelS = to_categorical(labelS, len(tokenizer.word_index) + 1)

    model = tensorflow.keras.models.Sequential([Embedding(input_dim = len(tokenizer.word_index) + 1, output_dim = 1024), 
                                                Bidirectional(LSTM(512)), Dense(len(tokenizer.word_index) + 1)])    
    model.compile(optimizer = Adam(), loss = CategoricalCrossentropy(from_logits = True), metrics = ['accuracy'])
    checkpoint = ModelCheckpoint(filepath = './models/best_text_generation.h5',
                                            save_weights_only = False,
                                            monitor = 'val_accuracy',
                                            mode = 'max',
                                            save_best_only = True,
                                            verbose = 1)
    history = model.fit(inputS, labelS, epochs = 1, batch_size = 32, validation_split = 0.3, callbacks = [checkpoint])

if __name__ == '__main__':
    train(padded_sequences)

