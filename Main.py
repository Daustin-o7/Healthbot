import User_Names

from plyer import notification
from ctypes import cdll
from http.client import responses
from pyexpat import model
from statistics import mode
from tkinter import W
from turtle import shape
from typing import List
import nltk
from tflearn import DNN 
#from keras import optimizer_v1
from keras.optimizer_v1 import Adam

from nltk.stem.lancaster import LancasterStemmer
from textblob import Word
stemmer = LancasterStemmer()

import numpy as np
import tensorflow as tf
import tflearn 
import random
import json 
import pickle


"""#For Nortification
def nortify(inp):
    if __name__=="__main__":
    
		    notification.notify(
			    title = "HEADING HERE",
			    message=" DESCRIPTION HERE" ,
		
			    # displaying time
			    timeout=2
)# waiting time
time.sleep(7)"""

with open("intents.json") as file:
    data = json.load(file)

try:
    with open("data.pickle", "rb") as f:
        words, labels, training, output = pickle.load(f)
except:
    words = []
    labels = []
    docs_x = []
    docs_y = []

    for intent in data["intents"]:
        for pattern in intent["patterns"]:
            wrds = nltk.word_tokenize(pattern)
            words.extend(wrds)
            docs_x.append(wrds)
            docs_y.append(intent["tag"])

        if intent["tag"] not in labels:
            labels.append(intent["tag"])

    words = [stemmer.stem(w.lower()) for w in words if w != "?"]
    words = sorted(list(set(words)))

    labels = sorted(labels)

    training = []
    output = []

    out_empty = [0 for _ in range(len(labels))]

    for x, doc in enumerate(docs_x):
        bag = []

        wrds = [stemmer.stem(w.lower()) for w in doc]

        for w in words:
            if w in wrds:
                bag.append(1)
            else:
                bag.append(0)

        output_row = out_empty[:]
        output_row[labels.index(docs_y[x])] = 1

        training.append(bag)
        output.append(output_row)


    training = np .array(training)
    output = np.array(output)

    with open("data.pickle", "wb") as f:
        pickle.dump((words, labels, training, output), f)

#model.DNN = sequential{}
#model = tflearn.DNN(net)

from tensorflow.python.framework import ops
ops.reset_default_graph()
net = tflearn.input_data(shape=[None, len(training[1])])
net = tflearn.fully_connected(net, 512)
net = tflearn.fully_connected(net, 512)
#net = tflearn.lstm(net,3)
#net = tflearn.dropout(net, 0.5)
net = tflearn.fully_connected(net, 512)
net = tflearn.fully_connected(net, len(output[1]), activation="Softmax")
net = tflearn.regression(net)
model = tflearn.DNN(net)
Adam(lr=0.001, beta_1=0.9, beta_2=0.999, epsilon=1e-08, decay=0.0)
try:
   model.load('model.tflearn')
except:
    
    # net = tflearn.input_data(shape=[None, len(training[1])])
    # net = tflearn.fully_connected(net, 512)
    # net = tflearn.fully_connected(net, 512)
    # #net = tflearn.lstm(net,3)
    # #net = tflearn.dropout(net, 0.5)
    # net = tflearn.fully_connected(net, 512)
    # net = tflearn.fully_connected(net, len(output[1]), activation="Softmax")
    # net = tflearn.regression(net)
    # model = tflearn.DNN(net)
    # Adam(lr=0.001, beta_1=0.9, beta_2=0.999, epsilon=1e-08, decay=0.0)
    
    model.fit(training, output, n_epoch=100, batch_size=64, show_metric=True)
    model.save('model.tflearn')
    #optimizers.adam_v2(lr=0.001, beta_1=0.9, beta_2=0.999, epsilon=1e-08, decay=0.0)
    #adam = tf.optimizers_v1.
    #Adam(0.001)
    
    #tf.losses('categorical_crossentropy')
    #model.compile(optimizer= Adam,losses='categorical_crossentropy',metrics=['accuracy'])
    
"""model.compile(optimizer=tf.keras.optimizers.Adam(learning_rate=1e-3),
              loss=tf.keras.losses.BinaryCrossentropy(),
              metrics=[tf.keras.metrics.BinaryAccuracy(),
                       tf.keras.metrics.FalseNegatives()])"""

    
    

def bag_of_words(s, words):
    bag = [0 for _ in range(len(words))]

    s_words = nltk.word_tokenize(s)
    s_words = [stemmer.stem(word.lower()) for word in s_words]

    for se in s_words:
        for i, w in enumerate(words):
            if w == se:
                bag[i] = 1
            
    return np.array(bag)


def chat():
    print("Let's start talk.\n * Enter the name of diseases to get common medicine,prevention and suggested food.\n * (type quit to stop)")
    while True:
        inp = input("You: ")
        if inp.lower() == "quit":
            break

        results = model.predict([bag_of_words(inp, words)])[0]
        results_index = np.argmax(results)
        tag = labels[results_index]


        for tg in data["intents"]:
            if tg['tag'] == tag:
                responses = tg['responses']
                
                
        print("Ishita: "+random.choice(responses))
        #print(results)

User_Names.write()
print("\n")
User_Names.read()


chat()