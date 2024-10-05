import numpy as np
import random
import pickle
import json
import tensorflow as tf
import nltk
from nltk.stem import WordNetLemmatizer
from tensorflow.keras.optimizers import SGD



lemmatizer = WordNetLemmatizer()

# Load the JSON file
with open(r'/Users/cansarma/Desktop/NullClass/NLP ChatBot/intents.json') as file:
    intents = json.load(file)

# Remember the steps of NLP
# 1. Text Preprocessing: To preprocess the data
words = []
classes = []
documents = []
ignoreLetters = ["!", "?", ".", ","]

# Tokenization
for intent in intents["intents"]:
    for pattern in intent["patterns"]:
        wordList = nltk.word_tokenize(pattern)
        words.extend(wordList)
        documents.append((wordList, intent["tag"]))  # This should be a tuple
        if intent["tag"] not in classes:
            classes.append(intent["tag"])

words = [lemmatizer.lemmatize(word.lower()) for word in words if word not in ignoreLetters]
words = sorted(set(words))
classes = sorted(set(classes))

pickle.dump(words, open("words.pkl", "wb"))
pickle.dump(classes, open("classes.pkl", "wb"))

training = []
outputEmpty = [0] * len(classes)

for document in documents:
    bag = []
    wordPatterns = document[0]  # Accessing the word patterns from the document tuple
    wordPatterns = [lemmatizer.lemmatize(word.lower()) for word in wordPatterns]
    for word in words:
        bag.append(1 if word in wordPatterns else 0)

    outputRow = list(outputEmpty)
    outputRow[classes.index(document[1])] = 1
    training.append(bag + outputRow)

random.shuffle(training)
training = np.array(training)

trainX = training[:, :len(words)]
trainY = training[:, len(words):]  # Corrected the slicing

# Build the model
model = tf.keras.Sequential()
model.add(tf.keras.layers.Dense(128, input_shape=(len(trainX[0]),), activation='relu'))
model.add(tf.keras.layers.Dropout(0.5))
model.add(tf.keras.layers.Dense(64, activation='relu'))
model.add(tf.keras.layers.Dropout(0.5))
model.add(tf.keras.layers.Dense(len(trainY[0]), activation='softmax'))

# Compile the model with loss function, optimizer, and metrics
sgd = SGD(learning_rate=0.01, momentum=0.9, nesterov=True)
model.compile(loss="categorical_crossentropy", optimizer=sgd, metrics=["accuracy"])

# Train the model using the training data
hist = model.fit(trainX, trainY, epochs=250, batch_size=5, verbose=1)

# Save the model
model.save("chatbot_model.h5", hist)
print("Done")











