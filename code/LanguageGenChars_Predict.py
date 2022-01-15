# -*- coding: utf-8 -*-
"""
Purpose: Character level Natural Language Generation (NLG). This file loads a
        previously trained character NLG model from LanguageGenChars_train.py, 
        and predicts subsequent chars.

To run: 
    1) Set constants below to be the same as the languagegenchars_train.py file
    2) At Anaconda command prompt enter
    >> python languagegenchars_predict.py

"""

# ---
# Libs

import numpy as np
import os
from datetime import datetime, timedelta

import re

import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow import keras
from tensorflow.keras.layers import LSTM, Dense, Dropout
from tensorflow.keras import utils as keras_utils


# ---
# Constants

# Set CURR_DIR to the subdir with this PY file. Everything else is relative to this subdir.
CURR_DIR = "C:\\MyPy\\LanguageGeneration\\NaturalLanguageGen"

# Predictions reuses the previously cleaned file.
INPUT_FILE = '.\\Data\\Complete_Shakespeare_cleaned.txt'

MODEL_WEIGHTS_FILE = ".\\Saved_Model\\training_GenChars2\\cp_Epoch_{epoch:02d}_Loss_{loss:.3f}.ckpt"
MODEL_WEIGHTS_DIR = os.path.dirname(MODEL_WEIGHTS_FILE)

MODEL_IMG_FILE = ".\\model_GenChars.png"
MODEL_RESULTS = ".\\model_GenChars_results.csv"

# The constants below MUST be the SAME as the model trained in LanguageGenChars_training.py.
#NUM_EPOCHS = 100
SEQ_LEN = 100
#BATCH_SIZE = 256
UNITS = 128

NUM_CHARS_PREDICT = 200

# ---
# Funcs

def clean_text(text):
    """
    Purpose: Pass a string, this func will remove everything and only leave 
            A-Z, a-z and sentence endings. It will also remove brackets [] and 
            everything between those brackets like [_Exit._], [_Exeunt._], etc.
    """

    # Remove brackets and the text within the brackets. 
    text = "".join(re.split("\(|\)|\[|\]", text)[::2])

    # Remove quotes and replace with no space. 
    text = re.sub(r"[\'\"\‘\’\`\ʹ]", "", text) 
   
    # Keep only a-z and sentence endings, everything else gets a space. 
    new_string = re.sub("[^a-zA-Z.?!;]", " ", text).strip()
    
    # Remove consective spaces and leave only one space.
    new_string = re.sub(" +", " ", new_string)
    
    new_string = new_string.lower()
            
    return(new_string)


# ---
# Main

start_time = datetime.now()

os.chdir(CURR_DIR)

# Load the previously cleaned file.
with open(INPUT_FILE, 'r', encoding='utf-8') as file:
    text = file.read()

# Load less data for optional model evaluations below. 
text = text[0:int(len(text)/4)]

# NOTE: No need to clean here since the previously cleaned TXT file from 
# the training file is reused here. 
# Clean the data. Since the NN will try to learn to predict the next character
# the fewer required characters it has to learn the model might be more accurate.
#text = clean_text(raw_text)

# Save the cleaned text to see the text the model used. 
#with open(OUTPUT_FILE, 'w', encoding='utf-8') as file:
#    file.write(text)

# NN's and other ML algorithms work with numeric data vs. text. Here set() 
# gets unique characters. Next, each unique character is assigned an integer
# in the order in which the characters were sorted. 
chars = sorted(list(set(text)))
char_num_map = dict((c, i) for i, c in enumerate(chars))

input_char_len = len(text)
vocab_len = len(chars)

print ("Total number of characters overall:", input_char_len)
print ("Total unique characters:", vocab_len)

x_seq_num = []
y_pred_num = []

# x_seq_num is a list of lists. The inner list is a sequence of SEQ_LEN. For 
# each input sequence, save a corresponding character to be predicted in
# y_pred_num[]. 
print("\nPreparing sequences.")
for i in range(0, input_char_len - SEQ_LEN, 1):
    
    # Define an input sequence of characters.  
    in_seq = text[i:i + SEQ_LEN]

    # Holds 1 predicted character for each sequence from in_seq. 
    out_seq = text[i + SEQ_LEN]

    # Convert each character in in_seq and out_seq to an integer. Re-use the 
    # char_num_map dict created earlier to find that char to integer mapping. 
    x_seq_num.append([char_num_map[char] for char in in_seq])
    y_pred_num.append(char_num_map[out_seq])
    
    
num_sequences = len(x_seq_num)
print ("\nNumber of sequences:", num_sequences)

# Numpy reshape will reshape x_seq_num to have samples, sequence length and 
# input dimensions. This input is expected by our NN. 
X = np.reshape(x_seq_num, (num_sequences, SEQ_LEN, 1))

# Normalize the integers to be within a range of zero to one. When a NN is fit 
# on scaled data that uses a small range of values (like zero to 1) the 
# network can be more effective learning the output. 
X = X/float(vocab_len)    
    
# One-hot encode the char numbers to be predicted. 
y = keras_utils.to_categorical(y_pred_num)

# Define the model. 
# Note, If no activation function is chosen it defaults to activation = 'tanh', 
# however added this param to be explicit. See model.get_config() output below 
# for details.
# Simple Model. 
# Must be the SAME as the model trained in LanguageGenChars_train.py.
model = Sequential()
model.add(LSTM(UNITS, activation='tanh', input_shape=(X.shape[1], X.shape[2]), return_sequences = False, name = "layer_1"))
model.add(Dropout(0.2, name = "layer_2"))
model.add(Dense(y.shape[1], activation='softmax', name = "layer_3"))


'''
# Deeper model. 100 sequence length, 3 LSTM layers, each having 700 units, trained 
# across 100 epochs. Takes 45 mins PER epoch on P3.2xlarge EC2 instance, very costly!
model = Sequential()
model.add(LSTM(UNITS, activation='tanh', input_shape=(X.shape[1], X.shape[2]), return_sequences=True))
model.add(Dropout(0.2))
model.add(LSTM(UNITS))
model.add(Dropout(0.2))
#model.add(LSTM(UNITS))
#model.add(Dropout(0.2))
model.add(Dense(y.shape[1], activation='softmax'))
'''

model.summary()

# Compile the model above. 
model.compile(loss = 'categorical_crossentropy', 
              optimizer='adam',
              #optimizer = RMSprop(learning_rate=0.01),  # Maybe try a different optimizer and learning rate. 
              metrics = ['accuracy'])

print("\nModel Config:\n", model.get_config() )

# Optional - Evaluate the Untrained model. 
print("\nEvaluating the untrained model...")
loss, acc = model.evaluate(X, y, verbose=2)
print("\nUntrained model accuracy: {:5.2f}%".format(100 * acc))

model_weights = tf.train.latest_checkpoint(MODEL_WEIGHTS_DIR) 
print("\nLoading best model weight file: %s" % model_weights)
model.load_weights(model_weights)

# Required - Re-evaluate the trained model to get it going before making predictions.
print("\nEvaluating the trained model...")
loss, acc = model.evaluate(X, y, verbose=2)
print("\nTrained model accuracy: {:5.2f}%".format(100 * acc))


# Make a prediction. 

num_to_char = dict((i, c) for i, c in enumerate(chars))

# x_seq_num is a list of lists. The inner list is numbers of SEQ_LEN long.
# Get a random starting point in the inner list for 1 numeric sequence to make 
# a prediction below. 
start = np.random.randint(0, len(x_seq_num) - 1)
sequence = x_seq_num[start]

print("\n-----\nTry predict method 1\n-----")
print("Random Seed:")
print("\"", ''.join([num_to_char[value] for value in sequence]), "\"")
print("\nNLG chars:")

gen_text = ""

for i in range(NUM_CHARS_PREDICT):
    
    # Reshape to samples, sequence length and input dimensions.
    x = np.reshape(sequence, (1, len(sequence), 1))
    
    # If the training file normalized the numbers between 0 to 1 then add that here.
    x = x / float(vocab_len)  
    
    prediction = model.predict(x, verbose=0)
    
    # Prediction is for all chars. The total chars is in input_char_len above. Need
    # to get the highest prediction with argmax. Next, convert that prediction
    # index location to the predicted char. 
    index = np.argmax(prediction)
    pred_char = num_to_char[index]
    
    # Save the generated text to print below. 
    gen_text = gen_text + pred_char

    # Add the argmax predicted index location to our sequence then truncate the
    # beginning of the sequence list by 1 so that the sequence list remains 
    # SEQ_LEN long. 
    sequence.append(index)
    sequence = sequence[1:len(sequence)]

print(gen_text)


start = np.random.randint(0, len(x_seq_num) - 1)
sequence = x_seq_num[start]

print("\n-----\nTry predict method 2\n-----")
print("Random Seed:")
print("\"", ''.join([num_to_char[value] for value in sequence]), "\"")
print("\nNLG chars:")

gen_text = ""

for i in range(NUM_CHARS_PREDICT):
    x = np.reshape(sequence, (1, len(sequence), 1))
    
    # If the training file normalized the numbers between 0 to 1 then add that here.
    x = x / float(vocab_len)  
    
    prediction = model.predict(x, verbose=0)
    
    # Predictions for each char in the vocabulary. With this prediction method, 
    # get a random prediction with random.choice(). Next, convert that 
    # prediction index location to the predicted char. 
    X = prediction[0] 
    index = np.random.choice(len(X), p=X)
    
    pred_char = num_to_char[index]
    
    # Save the generated text to print below. 
    gen_text = gen_text + pred_char

    # Add the argmax predicted index location to our sequence then truncate the
    # beginning of the sequence list by 1 so that the sequence list remains 
    # SEQ_LEN long. 
    sequence.append(index)
    sequence = sequence[1:len(sequence)]

print(gen_text)


# Print stats about the run.
end_time = datetime.now()
elapsed_time = end_time - start_time
time_diff_mins = elapsed_time / timedelta(minutes=1)
print("\nTotal runtime %.1f minutes or %.1f hours." % (time_diff_mins, time_diff_mins / 60))

