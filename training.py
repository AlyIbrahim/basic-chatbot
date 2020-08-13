import json, pickle, random
import nltk
from nltk.stem import WordNetLemmatizer
lemmatizer = WordNetLemmatizer()
#nltk.download('punkt')
#nltk.download('wordnet')

import numpy as np
#import tflearn
import tensorflow
import warnings

from keras.models import Sequential
from keras.layers import Dense, Activation, Dropout
from keras.optimizers import SGD

warnings.filterwarnings("ignore")

tensorflow.compat.v1.logging.set_verbosity(tensorflow.compat.v1.logging.ERROR)
#tensorflow.logging.set_verbosity(tensorflow.logging.ERROR)

words = []
classes = []
documents = []


intents_file = open('intents.json').read()
intents = json.loads(intents_file)['intents']

# Collect data (tokens, classes) from intents
for intent in intents:
  tag = intent['tag']
  classes.append(tag)
  patterns = intent['patterns']
  for pattern in patterns:
    tokenized_words = nltk.word_tokenize(pattern)
    words.extend(tokenized_words)
    documents.append((tag, tokenized_words))

# Data pre-processing
words = [ lemmatizer.lemmatize(w.lower()) for w in words ]
words = sorted(list(set(words)))
classes = sorted(classes)

# Saving Data Objects
pickle.dump(words, open('data/words.pkl', 'wb'))
pickle.dump(classes, open('data/classes.pkl', 'wb'))

#print("Words : ", words)
#print("Classes : ", classes)
#print("Documents : ", documents)

# Encoding data into bag of words
training_data = []
result = [ 0 for _ in range(len(classes)) ]

for tag, tokenized_words in documents:
#  print("Tokenized Words : ", tokenized_words)
  lemmatized_pattern= [lemmatizer.lemmatize(w.lower()) for w in tokenized_words]

  bag = [ 1 if word in lemmatized_pattern else 0 for word in words]
  result = [ 1 if tag == c else 0 for c in classes ]
  training_data.append((bag, result))

#  print("Bag : ", bag)
#print("Training : ", training_data)

# Building and training the model

training_data = np.array(training_data, dtype=object)
train_x = list(training_data[:,0])
train_y = list(training_data[:,1])

# TFLearn Model
#tensorflow.reset_default_graph()
#net = tflearn.input_data(shape=[None, len(train_x[0])])
#net = tflearn.fully_connected(net, 8)
#net = tflearn.fully_connected(net, 8)
#net = tflearn.fully_connected(net, len(train_y[0]), activation="softmax")
#net = tflearn.regression(net)
#model = tflearn.DNN(net)
#print(train_x)
#print(train_y)
#model.fit(train_x, train_y, n_epoch=1000, batch_size=8, show_metric=True)
#model.save("models/model.tflearn")
#model.load("models/model.tflearn")

# Keras Sequential Model with SGD
model = Sequential()
model.add(Dense(128, input_shape=(len(train_x[0]),), activation='relu'))
model.add(Dropout(0.5))
model.add(Dense(64, activation='relu'))
model.add(Dropout(0.5))
model.add(Dense(len(train_y[0]), activation='softmax'))
# Compile model. Stochastic gradient descent with Nesterov accelerated gradient gives good results for this model
sgd = SGD(lr=0.01, decay=1e-6, momentum=0.9, nesterov=True)
model.compile(loss='categorical_crossentropy', optimizer=sgd, metrics=['accuracy'])
#fitting and saving the model 
hist = model.fit(np.array(train_x), np.array(train_y), epochs=200, batch_size=5, verbose=1)
model.save('models/iago_model.h5', hist)


# Prediction
while True:
  input_line = input("You: ")
  if input_line.lower() == "quit":
    break

  in_words = nltk.word_tokenize(input_line)
  in_words = [ lemmatizer.lemmatize(w.lower()) for w in in_words  ]
#  print("Input : ",in_words)
#  print("Words : ",words)
  bag = np.array([ 1 if w in in_words else 0 for w in words ])
  bag = bag[None, :]
  result = model.predict(bag)
  result_ind = np.argmax(result)
  tag = classes[result_ind]
#  print("Bag : ", bag)
#  print("Prediction : ", tag)
  intents_file = open('intents.json').read()
  intents = json.loads(intents_file)['intents']
  for intent in intents:
    if tag == intent['tag']:
      responses = intent['responses']
      break
  print(random.choice(responses))

