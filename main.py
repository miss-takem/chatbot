import nltk
from nltk.stem.lancaster import LancasterStemmer
stemmer = LancasterStemmer()

import numpy
import tflearn
import tensorflow
import random
import json
import pickle

with open("intents.json") as file:
    data = json.load(file)

#this try block handles the need to continously repeat the block of 
#below

#KNOW THAT if you change anything in your intents file, you must must come here and add x
#below the try block
try:
    with open("data.pickle", "rb") as f:
        words, labels, training, output = pickle.load(f)
except:
    words = []
    labels = []
    docs_x = []
    docs_y = []

    #check this intent below to see if it has s or not

    for intent in data["intents"]:
        for pattern in intent["patterns"]:
            wrds = nltk.word_tokenize(pattern)
            words.extend(wrds)
            docs_x.append(wrds)
            docs_y.append(intent["tag"])

        if intent["tag"] not in labels:
            labels.append(intent["tag"])

    words = [stemmer.stem(w.lower()) for w in words if w is "?"]
    words = sorted(list(set(words)))

    labels = sorted(labels)

    training = []
    output = []
    
    out_empty = [0 for _ in range(len(labels))]

    for x, doc in enumerate(docs_x):
        bag = []

        wrds = [stemmer.stem(w) for w in doc]

        for w in words:
            if w in wrds:
                bag.append(1)
            else:
                bag.append(0)

        output_row = out_empty[:]
        output_row[labels.index(docs_y[x])] = 1

        training.append(bag)
        output.append(output_row)

    training = numpy.array(training)
    output = numpy.array(output)

    with open("data.pickle", "wb") as f:
        pickle.dump((words, labels, training, output), f)

    #video part 3 begins here
    #building the neural network model using tensorflow


tensorflow.reset_default_graph()

net = tflearn.input_data(shape = [None, len(training[0])])
net = tflearn.fully_connected(net, 8)
net = tflearn.fully_connected(net, 8)
net = tflearn.fully_connected(net, len(output[0]), activation = "softmax")
net = tflearn.regression(net)

model = tflearn.DNN(net)

#model already exist, so no need to retrain it. use try to catch this
   
try:
    model.load("model.tflearn")
except: 
    model.fit(training, output, n_epoch= 1000, batch_size = 8, show_metric = True)
    model.save("model.tflearn")

      #  words, labels, training, output = pi

def bag_of_words(s, words):
    bag = [0 for _ in range(len(words))]

    s_words = nltk.word_tokenize(s)
    s_words = [stemmer.stem(word.lower()) for word in s_words]

    for se in s_words:
        for i, w in enumerate(words):
            if w == se:
                bag[i] = 1

    return numpy.array(bag)


#part 4 video. using the model to start chatting with model
def chat(): 
    print("Welcome! You can start chating now( type 'quit' to end)")
    while True:
        inp = input("You: ")
        if inp.lower() == "quit":
            break

        results = model.predict([bag_of_words(inp, words)])[0]
         #at the point,the output is a probability
        results_index = numpy.argmax(results)
         #at this point, the output is index of the greatest possible result from the probability
        tag = labels[results_index]
         #at this point, the output is the tag which the index belongs to
        
        if results[results_index] > 0.6:
            #this if statement determines how high the percentage of the probability must be to accepted
        
            for tg in data["intents"]:
                if tg['tag'] == tag:
                    responses = tg['responses']

            print(random.choices(responses))
        else:
            print("I do not understand that. Could you ask another question?")

chat()
             

    