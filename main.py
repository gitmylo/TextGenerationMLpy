import tensorflow as tf
import keras as keras
import keras.layers as layers
import numpy as np
import matplotlib.pyplot as plt
import io
import os
import random
import signal
import time
import model as mod

m = mod.textModel()

maxlen = 40
step = 3

x = 0
y = 0
model = None

modelname = input("Modelname: ")
modelPath = "models/" + modelname

path = './data/' + input("Enter the name of the text file: ")
text = m.load_text(path)
chars = sorted(list(set(text)))
# cut the text in semi-redundant sequences of maxlen characters
(x, y) = m.cut_text(step, text)

if not os.path.exists(modelPath):
    print("Creating model")
    lstmCount = int(input("Enter the number of LSTM layers (rec: 128): "))
    model = m.get_model(lstmCount, maxlen)
else:
    print("Loading model")
    model = m.load_model(modelPath, "saved")

mode = -1
while mode == -1:
    mode = input("Mode (train, gen): ")
    if mode == "train":
        mode = 0
    elif mode == "gen":
        mode = 1
    else:
        mode = -1
        print("Invalid mode")

if mode == 0:

    def handler(signum, frame):
        print("Quitting, Saving model...")
        model.save_weights(modelPath + "/saved/data")
        print("Model saved. Exiting...")
        exit(0)

    signal.signal(signal.SIGINT, handler)

    model.save(modelPath)

    epochs = 999999
    batch_size = 128
    SaveEpochs = 100

    for epoch in range(int(epochs / SaveEpochs)):
        model.fit(x, y, batch_size=batch_size, epochs=SaveEpochs)
        print("Saving model...")
        model.save_weights(modelPath + "/saved/data")

elif mode == 1:
    while True:
        seed = input("Enter a seed: ")
        diversity = float(input("Enter a diversity (decimal number eg 0.2, 0.4, 1.2): "))
        seed = seed.lower()
        print(m.generate(seed, diversity))