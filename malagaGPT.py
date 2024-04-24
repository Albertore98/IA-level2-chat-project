import random
import json
import pickle
import numpy as np
import time
import nltk
from nltk.stem import WordNetLemmatizer

from keras.models import load_model


import tkinter as tk

lemmatizer = WordNetLemmatizer()

#Importamos los archivos generados en el código anterior
intents = json.loads(open('intents.json', encoding='utf-8').read())
words = pickle.load(open('words.pkl', 'rb'))
classes = pickle.load(open('classes.pkl', 'rb'))
model = load_model('chatbot_model.keras')

#Pasamos las palabras de oración a su forma raíz
def clean_up_sentence(sentence):
    sentence_words = nltk.word_tokenize(sentence)
    sentence_words = [lemmatizer.lemmatize(word) for word in sentence_words]
    return sentence_words

#Convertimos la información a unos y ceros según si están presentes en los patrones
def bag_of_words(sentence):
    sentence_words = clean_up_sentence(sentence)
    bag = [0]*len(words)
    for w in sentence_words:
        for i, word in enumerate(words):
            if word == w:
                bag[i]=1
    return np.array(bag)

#Predecimos la categoría a la que pertenece la oración
def predict_class(sentence):
    bow = bag_of_words(sentence)
    res = model.predict(np.array([bow]))[0]
    max_index = np.where(res ==np.max(res))[0][0]
    category = classes[max_index]
    return category

#Obtenemos una respuesta aleatoria
def get_response(tag, intents_json):
    list_of_intents = intents_json['intents']
    result = ""
    for i in list_of_intents:
        if i["tag"]==tag:
            result = random.choice(i['responses'])
            break
    return result

def set_response(input_entry, window, str_var):
    entry = input_entry.get()
    if entry == "":
        str_var.set("Escribe algo...")
    else:
        response = get_response(predict_class(entry), intents_json=intents)
        if response is None:
            return
        response_splitted = response.split(sep=" ")
        response_displayed = ""
        for res in response_splitted:
            response_displayed += res + " " 
            str_var.set(response_displayed)
            time.sleep(0.3)
            window.update()


#Ejecutamos el chat en bucle
if __name__ == "__main__":
    window = tk.Tk()
    window.title("Chatbot")
    window.geometry("600x800")
    tk.Label(text="MariCarmen:", font=("Arial", 20)).pack()
    
    input_entry = tk.Entry(width=80)
    
    str_var = tk.StringVar(value="Dime algo")
    output_entry = tk.Label(window, textvariable=str_var, wraplength=270)
    
    set_label_response = lambda: set_response(input_entry, window, str_var) 
    input_entry.bind('<Return>', lambda evt: set_label_response())
    
    input_entry.pack()
    tk.Button(text="Enviar", font=("Arial", 12), command=set_label_response).pack()
    tk.Label(text="IA:", font=("Arial", 16)).pack()
    output_entry.pack()

    window.mainloop()