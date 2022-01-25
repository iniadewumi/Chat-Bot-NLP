import json, pickle, nltk, pathlib
import numpy as np
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense,Dropout

HOME = pathlib.Path().resolve()
MODELS = HOME/"MODELS"
    
with open('DATASETS/training.pickle', 'rb') as f:
    inp = pickle.load(f)
    

X_train = inp["training"]
y_train = inp["output"]

model = Sequential(
    [Dense(128, activation='relu', input_shape=(len(X_train[0]),)),
    Dropout(0.25),
    Dense(64, activation='relu'),
    Dropout(0.25),
    Dense(len(y_train[0]), activation='softmax')]
    )

model.compile(loss='categorical_crossentropy', optimizer='adam', metrics=['accuracy'])

model.fit(X_train, y_train, epochs=35, batch_size=32, verbose=1)
model.save(MODELS/"chatbot_model.h5")