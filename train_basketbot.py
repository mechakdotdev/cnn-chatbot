# libraries
import random
from keras.optimizers import SGD
from keras.layers import Dense, Dropout
from keras.models import load_model
from keras.models import Sequential
import numpy as np
import pickle
import json
import nltk
from nltk.stem import WordNetLemmatizer
lemmatizer = WordNetLemmatizer()
nltk.download("punkt")
nltk.download("wordnet")

#class ModelTraining
# init file
words = []
classes = []
documents = []
flaggedText = ["!", "?", ",", "."] # don't pick these up
intentData = open("intents.json").read()
intents = json.loads(intentData)

for intent in intents["intents"]:
    for pattern in intent["patterns"]:
        word = nltk.word_tokenize(pattern)
        words.extend(word)

        # add docs in corpus
        documents.append((word, intent["tag"]))
        # add to list of classes
        if intent["tag"] not in classes:
            classes.append(intent["tag"])

# lemmatizer aka where we clean up
words = [lemmatizer.lemmatize(word.lower()) for word in words if word not in flaggedText]
words = sorted(list(set(words)))

classes = sorted(list(set(classes)))

print(len(documents), "documents")

print(len(classes), "classes", classes)

print(len(words), "unique lemmatized words", words)


pickle.dump(words, open("words.pkl", "wb"))
pickle.dump(classes, open("classes.pkl", "wb"))

# AC 2.1 - initializing training data
training = []
output_empty = [0] * len(classes)
for doc in documents:
    # bow initialisation
    bag = []
    # tokenized words
    pattern_words = doc[0]
    # find stem for each word
    pattern_words = [lemmatizer.lemmatize(word.lower()) for word in pattern_words]
    # if word is in pattern create bow w/ 1
    for word in words:
        bag.append(1) if word in pattern_words else bag.append(0)

    # in each pattern 0 represents a tag and 1 is the current tag
    output_row = list(output_empty)
    output_row[classes.index(doc[1])] = 1

    training.append([bag, output_row])

# shnp.array is for shuffled set of features
random.shuffle(training)
training = np.array(training)

# X - patterns
# Y - intents
train_x = list(training[:, 0])
train_y = list(training[:, 1])
print("Completed sets for training data.")

# AC 2.2 training to create dnn model
model = Sequential()
model.add(Dense(128, input_shape=(len(train_x[0]),), activation="relu"))
model.add(Dropout(0.5))
model.add(Dense(64, activation="relu"))
model.add(Dropout(0.5))
model.add(Dense(len(train_y[0]), activation="softmax"))
model.summary()

# submission 3 - Stochastic grad. so we can evaluate accuracy
sgd = SGD(lr=0.01, decay=1e-6, momentum=0.9, nesterov=True)
model.compile(loss="categorical_crossentropy", optimizer=sgd, metrics=["accuracy"])

# save as
hist = model.fit(np.array(train_x), np.array(train_y), epochs=200, batch_size=5, verbose=1)
model.save("dnn_model.h5", hist)
print("Completed saving a DNN Model.")
