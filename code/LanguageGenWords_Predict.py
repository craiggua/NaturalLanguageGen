# -*- coding: utf-8 -*-
"""
Purpose: Word level Natural Language Generation (NLG). This file loads a
        previously trained word NLG model from LanguageGenChars_train.py, 
        and predicts subsequent words. 

To run: 
    1) Set constants below to be the same as the languagegenwords_train.py file
    2) At Anaconda command prompt enter
    >> python languagegenwords_predict.py

"""

# ---
# Libs

import os
from datetime import datetime, timedelta

import re
import numpy as np
from nltk import tokenize

import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Embedding, LSTM, Dense, Dropout, Bidirectional
from tensorflow.keras.preprocessing.sequence import pad_sequences
from tensorflow.keras.preprocessing.text import Tokenizer
from tensorflow.keras import utils as keras_utils

#import matplotlib.pyplot as plt


# ---
# Constants

# Set CURR_DIR to the subdir with this PY file. Everything else is relative to this subdir.
CURR_DIR = "C:\\MyPy\\LanguageGeneration\\NaturalLanguageGen"

# Predictions reuses the previously cleaned file.
INPUT_FILE = '.\\Data\\Complete_Shakespeare_cleaned.txt'

MODEL_WEIGHTS_FILE = ".\\Saved_Model\\training_GenWords2\\cp_Epoch_{epoch:02d}_Loss_{loss:.3f}.ckpt"
MODEL_WEIGHTS_DIR = os.path.dirname(MODEL_WEIGHTS_FILE)

MODEL_IMG_FILE = ".\\model_GenWords.png"
MODEL_RESULTS = ".\\model_GenWords_results.csv"

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


def generate_text(seed_text, max_words, max_sequence_len, model, tokenizer):
    """
    Purpose: Given a previously trained NLG model trained on words, pass a 
    string of seed text and other params to predict new text. 
            
    Created this function from the code below. 
    
    Source: https://colab.research.google.com/github/tensorflow/examples/blob/master/courses/udacity_intro_to_tensorflow_for_deep_learning/l10c03_nlp_constructing_text_generation_model.ipynb#scrollTo=DC7zfcgviDTp&line=1&uniqifier=1
            Apache License 2.0.
    """
    
    # Clean and lowercase the seed_text so it's like the text used in training.
    seed_text = clean_text(seed_text)
    
    # Convert seed text to a list, add padding, predict the next word, add
    # that predicted word to the end of the seed text string and repeat. 
    for _ in range(max_words):
    	token_list = tokenizer.texts_to_sequences([seed_text])[0]
    	token_list = pad_sequences([token_list], maxlen=max_sequence_len-1, padding='pre')
    	prediction = np.argmax(model.predict(token_list), axis=-1)
    	pred_word = ""
        
    	for word, index in tokenizer.word_index.items():
    		if index == prediction:
    			pred_word = word
    			break
            
    	seed_text += " " + pred_word
        
    return(seed_text)

# ---
# Main

start_time = datetime.now()

os.chdir(CURR_DIR)

# Load the previously cleaned file.
with open(INPUT_FILE, 'r', encoding='utf-8') as file:
    text = file.read()


# NOTE: No need to clean here since the previously cleaned TXT file from 
# the training file is reused here. 
# Clean the data and lowercase to reduce the number of tokens in the vocabulary. 
#text = clean_text(raw_text)

# Save the cleaned text to see the text the model used. 
#with open(OUTPUT_FILE, 'w', encoding='utf-8') as file:
#    file.write(text)

# Split by sentence endings. This will allow us to dynamically determine the 
# sequence length.
tok_sents = tokenize.sent_tokenize(text)

tokenizer = Tokenizer()
tokenizer.fit_on_texts(tok_sents)

# Get the total number of all words in the doc... just some stats about the data.
total_word_count = 0
for key, value in tokenizer.word_counts.items(): 
    total_word_count = total_word_count + value

# Each unique word in the input_file will be given a unique integer. This is 
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
# lengths. Therefore, need to pad the sequences so they have equal lengths. 

# Find the length of the longest sequence. Since this is a dymamic value driven 
# by the input file, it could be a malformed input file so ensure that any single
# sequence doesn't go beyond the constant MAX_SEQ_LEN above. Below pad_sequences() 
# will trucate sequences if necessary. 
max_sequence_len = max([len(x) for x in sequences])

if max_sequence_len > MAX_SEQ_LEN:
    max_sequence_len = MAX_SEQ_LEN

# https://www.tensorflow.org/api_docs/python/tf/keras/preprocessing/sequence/pad_sequences
# To make each sequence the same length, add padding at the end of the sequence. 
# Also, truncate sequences from the end if they that exceeed the maxlen. 
sequences = np.array(pad_sequences(sequences, maxlen = max_sequence_len, padding='pre', truncating = 'pre'))

print ("\nNumber of patterns:", len(sequences))

# A given set of sequence indicies in x_seq_num, should predict a particular 
# index stored in y_pred_num.
x_seq_num, y_pred_num = sequences[:,:-1],sequences[:,-1]

# One-hot encode the words to be predicted. 
y = keras_utils.to_categorical(y_pred_num, num_classes = unique_word_cnt)

input_len = max_sequence_len - 1    


# A simple model. 
model = Sequential()
model.add(Embedding(unique_word_cnt, OUTPUT_DIM, input_length = input_len, name = "layer_1"))
model.add(LSTM(UNITS, name = "layer_2"))
model.add(Dropout(0.2, name = "layer_3"))
model.add(Dense(unique_word_cnt, activation ='softmax', name = "layer_4"))

'''
# More advanced model. 
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

# Optional: Evaluate the Untrained model. 
print("\nEvaluating the untrained model...")
loss, acc = model.evaluate(x_seq_num, y, verbose=2)
print("\nUntrained model accuracy: {:5.2f}%".format(100 * acc))

# Load the best performing model. 
model_weights = tf.train.latest_checkpoint(MODEL_WEIGHTS_DIR) 
print("\nLoading best model weight file: %s" % model_weights)
model.load_weights(model_weights)

# Required - Re-evaluate the model. 
print("\nEvaluating the trained model...")
loss, acc = model.evaluate(x_seq_num, y, verbose=2)
print("\nRestored model accuracy: {:5.2f}%".format(100 * acc))

# Now, predict new text. Real Shakespeare in seed_text. 
#seed_text = "from fairest creatures we desire increase that thereby beauty rose might never die"
seed_text = "to be or not to be that is the question"

text = generate_text(seed_text, 20, max_sequence_len, model, tokenizer)
print("\nSeed text:\n", seed_text)
print("\nGenerated text:\n", text)

# Print stats about the run.
end_time = datetime.now()
elapsed_time = end_time - start_time
time_diff_mins = elapsed_time / timedelta(minutes=1)
print("\nTotal runtime %.1f minutes or %.1f hours." % (time_diff_mins, time_diff_mins / 60))


