import os

import tensorflow as tf
import keras as keras
import keras.layers as layers
import numpy as np
import matplotlib.pyplot as plt
import io
import random

class textModel:

    def __init__(self):
        self.char_indices = {}
        self.indices_char = {}
        self.text = ""
        self.load_known_chars()
        self.model = None
        self.maxlen = 40
        print('total chars:', len(self.chars))
        self.char_indices = dict((c, i) for i, c in enumerate(self.chars))
        self.indices_char = dict((i, c) for i, c in enumerate(self.chars))
        config = tf.ConfigProto()
        config.gpu_options.allow_growth = True
        sess = tf.Session(config=config)

    def setVars(self, maxlenNew):
        self.maxlen = maxlenNew

    def load_known_chars(self):
        if not os.path.isfile("dict.txt"):
            self.chars = "abcdefghijklmnopqrstuvwxyz"
            self.save_known_chars(self.chars)
        with open("dict.txt", "r", encoding="utf-8") as f:
            self.chars = f.read()

    def save_known_chars(self, chars):
        with open("dict.txt", "w+", encoding="utf-8") as f:
            f.write(chars)
        return chars

    def load_text(self, textFile):
        with io.open(textFile, encoding="utf-8") as f:
            self.text = f.read().lower()
        print('corpus length:', len(self.text))
        self.chars = self.save_known_chars(self.chars +
                                           self.string_difference(
                                               self.chars,
                                               self.text))
        print('total chars:', len(self.chars))
        return self.text

    def cut_text(self, step, text):
        self.sentences = []
        self.next_chars = []
        for i in range(0, len(text) - self.maxlen, step):
            self.sentences.append(text[i: i + self.maxlen])
            self.next_chars.append(text[i + self.maxlen])
        print("Number of sequences:", len(self.sentences))

        x = np.zeros((len(self.sentences), self.maxlen, len(self.chars)), dtype=np.bool)
        y = np.zeros((len(self.sentences), len(self.chars)), dtype=np.bool)
        for i, sentence in enumerate(self.sentences):
            for t, char in enumerate(sentence):
                x[i, t, self.char_indices[char]] = 1
            y[i, self.char_indices[self.next_chars[i]]] = 1
        return x, y

    def get_model(self, lstmCount, maxlen):
        self.model = keras.Sequential(
            [
                keras.Input(shape=(maxlen, len(self.chars))),
                layers.LSTM(lstmCount),
                layers.Dense(len(self.chars), activation="softmax"),
            ]
        )
        optimizer = keras.optimizers.RMSprop(learning_rate=0.01)
        self.model.compile(loss="categorical_crossentropy", optimizer=optimizer)
        return self.model

    def load_model(self, modelPath, data_name):
        self.model = keras.models.load_model(modelPath)
        self.model.load_weights(modelPath + "/" + data_name + "/data")
        return self.model

    def sample(self, preds, temperature=1.0):
        # helper function to sample an index from a probability array
        preds = np.asarray(preds).astype("float64")
        preds = np.log(preds) / temperature
        exp_preds = np.exp(preds)
        preds = exp_preds / np.sum(exp_preds)
        probas = np.random.multinomial(1, preds, 1)
        return np.argmax(probas)

    def generate(self, seed, diversity):
        generated = ""
        print('...Generating with seed: "' + seed + '"')

        for i in range(400):
            x_pred = np.zeros((1, self.maxlen, len(self.chars)))
            for t, char in enumerate(seed):
                x_pred[0, t, self.char_indices[char]] = 1.0
            preds = self.model.predict(x_pred, verbose=0)[0]
            next_index = self.sample(preds, diversity)
            next_char = self.indices_char[next_index]
            seed = seed[1:] + next_char
            generated += next_char
        return generated

    def string_difference(self, a, b):
        return "".join(list(set(b) - set(a)))