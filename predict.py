import random, pickle, json, logging, re

import numpy as np
import nltk
from nltk.stem import WordNetLemmatizer
lemmatizer = WordNetLemmatizer()
#import tflearn
from keras.models import load_model
#import tensorflow as tf

words =  pickle.load(open('data/words.pkl', 'rb'))
classes = pickle.load(open('data/classes.pkl', 'rb'))

model = load_model('models/iago_model.h5')

def main():
  while True:
    input_line = input("You: ")
    if input_line.lower() == "quit":
      break

    in_words = nltk.word_tokenize(input_line)
    in_words = [ lemmatizer.lemmatize(w.lower()) for w in in_words  ]
#    print("Input : ",len(in_words))
#    print("Words : ",words)

    bag = np.array([ 1 if w in in_words else 0 for w in words ])
    bag = bag[None, :]
#    print("Bag : ", bag.shape)
    result = model.predict(bag)
    result_ind = np.argmax(result)
    tag = classes[result_ind]
#    print(tag)
    intents_file = open('intents.json').read()
    intents = json.loads(intents_file)['intents']
    for intent in intents:
      if tag == intent['tag']:
        responses = intent['responses']
        break
    print(random.choice(responses))

if __name__=="__main__":
    main()

