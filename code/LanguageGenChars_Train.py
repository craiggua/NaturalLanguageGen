# -*- coding: utf-8 -*-
"""
Purpose: Character level Natural Language Generation (NLG). 
        a. This file trains a character based NLG model. 
        b. Model training has early stopping to prevent over-training.
        c. Save the best performing models during training.
        d. Metrics from each trained epoch are written to a CSV for later analysis
        e. Image of the neural network model.
        f. Information about the complete test run is saved (ex: number of epochs, 
        total run time, details about the model, etc.)
        g. Separate code to load the best performing model and generate text. 
        This saves time since it could take many hours to create a good 
        NLG model.

To run: 
    1) Set constants below. 
    2) At Anaconda command prompt enter
    >> python languagegenchars_train.py

"""

# ---
# Libs

import numpy
import os
from datetime import datetime, timedelta

import re

from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import LSTM, Dense, Dropout
from tensorflow.keras.callbacks import CSVLogger, ModelCheckpoint, EarlyStopping
from tensorflow.keras import utils as keras_utils

import matplotlib.pyplot as plt

# ---
# Constants

# Set CURR_DIR to the subdir with this PY file. Everything else is relative to this subdir.
CURR_DIR = "C:\\MyPy\\LanguageGeneration\\NaturalLanguageGen"

# Path to the input text file. Also, name the output file and choose a destination.
INPUT_FILE = '.\\Data\\Complete_Shakespeare_Copy.txt'
OUTPUT_FILE = '.\\Data\\Complete_Shakespeare_cleaned.txt'

MODEL_WEIGHTS_FILE = ".\\Saved_Model\\training_GenChars2\\cp_Epoch_{epoch:02d}_Loss_{loss:.3f}.ckpt"
MODEL_WEIGHTS_DIR = os.path.dirname(MODEL_WEIGHTS_FILE)

MODEL_IMG_FILE = ".\\model_GenChars_deepmodel.png"
MODEL_RESULTS = ".\\model_GenChars_results_deepmodel.csv"

NUM_EPOCHS = 2
SEQ_LEN = 100
BATCH_SIZE = 256
UNITS = 128


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


def plot_graph(history, x_label, string):
    """
    Purpose: Send the Keras model.fit history results (keras.callbacks.History). 
             Send the key for the call back history dictionary (ex: 'loss'). A 
             graph plot will be created showing the key results for each epoch. 
    """    

    plt.plot(history.history[string])
    plt.xlabel(x_label)
    plt.ylabel(string)
    plt.ylim(ymin = 0)
    plt.show()


# ---
# Main

start_time = datetime.now()

os.chdir(CURR_DIR)

with open(INPUT_FILE, 'r', encoding='utf-8') as file:
    raw_text = file.read()
    
# Clean the data. Since the NN will try to learn to predict the next character
# the fewer required characters it has to learn the model might be more accurate.
text = clean_text(raw_text)

# Save the cleaned text to see the text the model actually used. 
with open(OUTPUT_FILE, 'w', encoding='utf-8') as file:
    file.write(text)


# NN's and other ML algorithms work with numeric data vs. text. Here set() 
# gets unique characters. Next, each unique character is assigned an integer
# in the order in which the characters were sorted. 
chars = sorted(list(set(text)))
char_num_map = dict((c, i) for i, c in enumerate(chars))

input_char_len = len(text)
vocab_len = len(chars)

print("Total number of characters overall:", input_char_len)
print("Total unique characters:", vocab_len)
print(char_num_map)

x_seq_num = []
y_pred_num = []

# x_seq_num is a list of lists. The inner list is a sequence of SEQ_LEN. For 
# each input sequence, save a corresponding integer in y_pred_num[] to be 
# predicted.
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
X = numpy.reshape(x_seq_num, (num_sequences, SEQ_LEN, 1))

# Normalize the integers to be within a range of zero to one. When a NN is fit 
# on scaled data that uses a small range of values (like zero to 1) the 
# network can be more effective learning the output. 
X = X/float(vocab_len)

# One-hot encode the chars to be predicted. 
y = keras_utils.to_categorical(y_pred_num)

# Define the model. 
# Note, if no activation function is chosen it defaults to activation = 'tanh', 
# however added this param to be explicit. See model.get_config() output below 
# for details.
# Simple Model. 
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
model.add(LSTM(UNITS, return_sequences=True))
model.add(Dropout(0.2))
model.add(LSTM(UNITS))
model.add(Dropout(0.2))
model.add(Dense(y.shape[1], activation='softmax'))
'''

model.summary()

# Compile the model above. 
model.compile(loss = 'categorical_crossentropy', 
              optimizer='adam',
              #optimizer = keras.optimizers.RMSprop(learning_rate=0.01),  # Try a different optimizer and learning rate. 
              metrics = ['accuracy'])


print("\nModel Config:", model.get_config())

# Create an image describing the NN.
keras_utils.plot_model(model, to_file = MODEL_IMG_FILE, show_shapes = True)

# CSVLogger: Log the model results per epoch to a CSV file. 
log_results = CSVLogger(MODEL_RESULTS, append = False)

# ModelCheckpoint: Save the best performing model weights.
checkpoint = ModelCheckpoint(filepath = MODEL_WEIGHTS_FILE, monitor='loss', verbose=1, save_weights_only=True, save_best_only=True, mode='min')

# EarlyStopping: Stop model training early if the loss doesn't move much.
# Can monitor accuracy or loss. 
early_stop = EarlyStopping(monitor = 'loss', patience = 5) 

# Collect callback names here for model.fit() below. 
callback_names = [log_results, checkpoint, early_stop]

# Train the model. 
history = model.fit(X, y, epochs = NUM_EPOCHS, batch_size = BATCH_SIZE, callbacks = callback_names, verbose = 1)

# Print stats about the run.
print("\nEpochs: %i \nBatch Size: %i \nUnits: %i" % (NUM_EPOCHS, BATCH_SIZE, UNITS))

end_time = datetime.now()
elapsed_time = end_time - start_time
time_diff_mins = elapsed_time / timedelta(minutes=1)

print("Total runtime %.1f minutes or %.1f hours." % (time_diff_mins, time_diff_mins / 60))
print("See %s for an image of the NN." % MODEL_IMG_FILE)
print("See %s for detailed model results." % MODEL_RESULTS)

# Plot and print model results.
for key in history.history.keys():
    plot_graph(history, "Epochs", key)
    
    print("\n%s" % key)
    for value in history.history[key]:
        print("%0.3f" % value)         

print("\n")

# NOTE: See the LanguageGenChars_Predict.py file to make predictions. 


