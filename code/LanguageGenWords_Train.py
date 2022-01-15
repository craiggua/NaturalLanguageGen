# -*- coding: utf-8 -*-
"""
Purpose: Word level Natural Language Generation (NLG). 
        a. This file trains a word based NLG model. 
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
    >> python languagegenwords_train.py

"""

# ---
# Libs

import os
from datetime import datetime, timedelta

import re
import numpy as np
from nltk import tokenize

from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Embedding, LSTM, Dense, Dropout, Bidirectional
from tensorflow.keras.preprocessing.sequence import pad_sequences
from tensorflow.keras.preprocessing.text import Tokenizer
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

MODEL_WEIGHTS_FILE = ".\\Saved_Model\\training_GenWords2\\cp_Epoch_{epoch:02d}_Loss_{loss:.3f}.ckpt"
MODEL_WEIGHTS_DIR = os.path.dirname(MODEL_WEIGHTS_FILE)

MODEL_IMG_FILE = ".\\model_GenWords.png"
MODEL_RESULTS = ".\\model_GenWords_results.csv"

NUM_EPOCHS = 2
MAX_SEQ_LEN = 160
BATCH_SIZE = 256
UNITS = 128
OUTPUT_DIM = 32


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
             Send the key for the call back history dictionary (ex 'loss'). A 
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

# Clean the data and lowercase to reduce the number of words in the vocabulary. 
text = clean_text(raw_text)

# Save the cleaned text to see the text the model used. 
with open(OUTPUT_FILE, 'w', encoding='utf-8') as file:
    file.write(text)

# Split by sentence endings. This will allow us to dynamically determine the 
# sequence length.
tok_sents = tokenize.sent_tokenize(text)

tokenizer = Tokenizer()
tokenizer.fit_on_texts(tok_sents)

# Get the total number of words in the cleaned text... just some stats about the data.
total_word_count = 0
for key, value in tokenizer.word_counts.items(): 
    total_word_count = total_word_count + value

# Each unique word in the cleaned text will be given a unique integer. This is 
# the size of our vocabulary. 
# Example:
#print(tokenizer.word_index)
unique_word_cnt = len(tokenizer.word_index) + 1

print ("Total number of all words:", total_word_count)
print ("Total unique words:", unique_word_cnt)

# Each text sequence is converted to a sequence of unique integers. The 
# tokenizer.word_index dictionary is used to create this mapping.
sequences = tokenizer.texts_to_sequences(tok_sents)

# Since the text was split into sentences, most sentences will have different
# lengths. Therefore, we need to pad the sequences so they have equal lengths. 
# Find the length of the longest sequence. Since this is a dymamic value driven 
# by the input file, it could be a malformed input file so ensure that any single
# sequence doesn't go beyond the constant MAX_SEQ_LEN above. Below pad_sequences()
# will trucate sequences if necessary. 
max_sequence_len = max([len(x) for x in sequences])

if max_sequence_len > MAX_SEQ_LEN:
    max_sequence_len = MAX_SEQ_LEN

# To make each sequence the same length, add padding at the end of the sequence. 
# Also, truncate sequences from the end if they that exceeed the maxlen. 
# For more info see: https://www.tensorflow.org/api_docs/python/tf/keras/preprocessing/sequence/pad_sequences
sequences = np.array(pad_sequences(sequences, maxlen = max_sequence_len, padding='pre', truncating = 'pre'))

print ("\nNumber of patterns:", len(sequences))

# A given set of sequence indicies in x_seq_num, should predict a particular 
# index stored in y_pred_num.
x_seq_num, y_pred_num = sequences[:,:-1],sequences[:,-1]

# One-hot encode the words to be predicted in y_pred_num.
y = keras_utils.to_categorical(y_pred_num, num_classes = unique_word_cnt)

input_len = max_sequence_len - 1    


# A simple model. 
model = Sequential()
model.add(Embedding(unique_word_cnt, OUTPUT_DIM, input_length = input_len, name = "layer_1"))
model.add(LSTM(UNITS, name = "layer_2"))
model.add(Dropout(0.2, name = "layer_3"))
model.add(Dense(unique_word_cnt, activation ='softmax', name = "layer_4"))

'''
# More advanced model. Train for 200+ epochs.
model = Sequential()
model.add(Embedding(unique_word_cnt, OUTPUT_DIM, input_length = input_len))
model.add(Bidirectional(LSTM(UNITS)))
model.add(Dense(unique_word_cnt, activation='softmax'))
'''

model.summary()

# Compile the model above.
model.compile(loss = 'categorical_crossentropy', 
              optimizer = 'adam',
              #optimizer = RMSprop(learning_rate=0.01),  # Maybe try a different optimizer and learning rate. 
              metrics = ['accuracy'])

# Optional print to see details of the model for logging and comparisons. 
print("\nModel Config:\n%s\n" % model.get_config())

# Create an image describing the NN.
keras_utils.plot_model(model, to_file = MODEL_IMG_FILE, show_shapes = True)

# CSVLogger - Log the model results per epoch to a CSV file. 
log_results = CSVLogger(MODEL_RESULTS, append = False)

# ModelCheckpoint - Save the best performing models.
checkpoint = ModelCheckpoint(filepath = MODEL_WEIGHTS_FILE, monitor='loss', save_weights_only=True, save_best_only=True, mode='min', verbose=1)

# EarlyStopping - Stop model training early if the loss doesn't move much.
# monitor can be accuracy or loss. 
early_stop = EarlyStopping(monitor = 'loss', patience = 5) 

callback_names = [log_results, checkpoint, early_stop]

# Train the model. 
history = model.fit(x_seq_num, y, epochs = NUM_EPOCHS, batch_size = BATCH_SIZE, callbacks = callback_names, verbose = 1)

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

# NOTE: See the LanguageGenWords_Predict.py file to make predictions. 

