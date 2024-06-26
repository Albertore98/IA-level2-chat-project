import random
import json
import pickle
import numpy as np

import nltk
from tensorflow import keras
from keras.layers import Dense, Dropout
from keras.models import Sequential
from nltk.stem import WordNetLemmatizer
from keras.optimizers import SGD
#from tensorflow.keras.models import Sequential
#from tensorflow.keras.layers import Dense, Activation, Dropout
#from tensorflow.keras.optimizers import SGD 

def words_to_bin(words, classes, documents, lemmatizer, training):
    output_empty = [0] * len(classes)
    max_words_length = len(words)

    for document in documents:
        bag = []
        word_patterns = document[0]
        word_patterns = [lemmatizer.lemmatize(word.lower()) for word in word_patterns]

        # Crear la bolsa de palabras
        for word in words:
            bag.append(1) if word in word_patterns else bag.append(0)

        # Asegurarse de que la bolsa de palabras tenga la misma longitud
        if len(bag) < max_words_length:
            bag += [0] * (max_words_length - len(bag))
        elif len(bag) > max_words_length:
            bag = bag[:max_words_length]

        # Crear la etiqueta de salida
        output_row = list(output_empty)
        output_row[classes.index(document[1])] = 1

        training.append([bag, output_row])

def classify_patterns(words, classes, documents, intents):
    for intent in intents['intents']:
        for pattern in intent['patterns']:
            word_list = nltk.word_tokenize(pattern)
            words.extend(word_list)
            documents.append((word_list, intent["tag"]))
            if intent["tag"] not in classes:
                classes.append(intent["tag"])

def build_model(train_x, train_y):
    model = Sequential()
    model.add(Dense(128, input_shape=(len(train_x[0]),), activation='relu'))
    model.add(Dropout(0.5))
    model.add(Dense(64, activation='relu'))
    model.add(Dropout(0.5))
    model.add(Dense(len(train_y[0]), activation='softmax'))
    return model

if __name__ == "__main__":
    words = []
    classes = []
    documents = []
    ignore_letters = ['?', '!', '¿', '.', ',', ';']
    lemmatizer = WordNetLemmatizer()

    try:
        intents = json.loads(open('intents.json', encoding="utf-8").read())
    except FileNotFoundError:
        print("ERROR: 'intents.json' not found, create a file 'intents.json' before starting the training.")
        exit(1)
    nltk.download('punkt', quiet=True)
    nltk.download('wordnet', quiet=True)

    # Clasifica los patrones y las categorías
    classify_patterns(words, classes, documents, intents)

    words = [lemmatizer.lemmatize(word) for word in words if word not in ignore_letters]
    words = sorted(set(words))

    pickle.dump(words, open('words.pkl', 'wb'))
    pickle.dump(classes, open('classes.pkl', 'wb'))

    # Pasa la información a unos y ceros según las palabras presentes en cada categoría para hacer el entrenamiento
    training = []
    words_to_bin(words, classes, documents, lemmatizer, training)

    # Convertir a listas regulares
    train_x = [item[0] for item in training]
    train_y = [item[1] for item in training]

    # Convertir a arrays de NumPy
    train_x = np.array(train_x)
    train_y = np.array(train_y)

    # Creamos la red neuronal
    model = build_model(train_x, train_y)

    # Utilizo 'tensorflow.keras.optimizers.SGD' en lugar de 'keras.optimizers.SGD'
    sgd = SGD(learning_rate=0.01, momentum=0.9, nesterov=True)
    model.compile(loss='categorical_crossentropy', optimizer=sgd, metrics=['accuracy'])

    train_process = model.fit(train_x, train_y, epochs=100, batch_size=5, verbose=1)

    # Se guarda el archivo del modelo en "chatbot_model.keras"
    model.save("chatbot_model.keras")