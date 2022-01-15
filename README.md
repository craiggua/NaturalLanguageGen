# NaturalLanguageGen
Character and word level natural language generation (NLG) tutorial. Uses Python and several types of neural networks (LSTM, deep LSTM, Bidirectional LSTM).

The code examples include the following: 
1. Character and word level NLG models
2. Model training has early stopping to prevent over-training
3. Save the best performing models during training 
4. Metrics, such as accuracy and loss, from each trained epoch are written to a CSV file for later analysis
5. Image of the neural network model
6. Information about the complete test run (ex: number of epochs, total run time, details about the model, etc.)
7. Separate code to load the best performing model and generate text. This saves time since it could take many hours to create a good NLG model. You may want to run the best model several times to get different and often amusing results. 

Note, separately 4, 5 and 6 above can also be done in [TensorBoard](https://www.tensorflow.org/tensorboard/get_started).

A few sample outputs:
![A few sample outputs](https://github.com/craiggua/NaturalLanguageGen/blob/main/img/LanguageGenChars_simplemodel_results.PNG)
