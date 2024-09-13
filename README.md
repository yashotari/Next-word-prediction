# Next Word Prediction Model

## Overview
This repository contains a Next Word Prediction model designed to predict the next word in a sentence or phrase based on historical patterns in a given text dataset. The project leverages Natural Language Processing (NLP) techniques and deep learning models to build an effective language model.

## Files
next word prediction.ipynb: Jupyter Notebook containing the code for data preprocessing, model building, training, and evaluation.
README.md: Documentation and instructions for the project.
US_Crime_Data.csv: The dataset used to train the model, which includes large text data for learning language patterns.

## Objective
The goal of this project is to create a model that can predict the next word in a sentence based on a given sequence of words. This type of predictive model has applications in text generation, autocomplete features, and smart suggestions in messaging platforms or text editors.

## Instructions
1. The dataset used for training the model is provided in the US_Crime_Data.csv file. This file contains a large collection of text data used for learning word sequences and language structure.

2. The next word prediction.ipynb Jupyter Notebook covers the following steps:

- **Data Preprocessing**: Tokenization of text,Removing unnecessary characters like punctuation and special symbols, creation of input sequences.
- **Model Building**: Uses an LSTM (Long Short-Term Memory) network for the next word prediction task.
- **Training**: The model is trained using the text data, learning to predict the next word based on previous word sequences.
- **Evaluation**: The model’s accuracy is evaluated to check its performance on unseen data.

3. Once the model is trained,test the model's predictions in real-time. You can input a sentence fragment, and the model will suggest the next word.

## Key Features
LSTM Model: The project uses an LSTM neural network, which is highly effective for sequence prediction tasks.
Custom Text Data: The model can be trained on any text data to adapt to the specific language style and patterns of that corpus.
Real-time Prediction: The prediction script allows you to input any sentence fragment and generate real-time next word suggestions.

## Dependencies
To run the project, ensure the following libraries are installed:

- pandas
- numpy
- keras
- tensorflow
- nltk


## Software Requirements
- **next word prediction.ipynb**: Requires Jupyter Notebook with Python and the necessary libraries.
- **US_Crime_Data.csv**: The dataset used to train the model, which includes large text data.

## Future Work
- **Model Optimization**: Experiment with different architectures (e.g., transformers) to improve accuracy.
- **Dataset Expansion**: Incorporate larger or more domain-specific datasets for training the model.
- **Multilingual Support**: Extend the model to support next word prediction in multiple languages.

## Collaboration Expectations
- Contributions and feedback are welcome through issues and pull requests.
- Please follow the repository’s contribution guidelines for adding improvements or new features.

Feel free to contact me for any questions or suggestions!


