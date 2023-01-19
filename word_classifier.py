import pickle
from tensorflow.keras.preprocessing.text import Tokenizer
from tensorflow.keras.preprocessing.sequence import pad_sequences
import numpy as np
import tensorflow as tf

def get_sequences(tokenizer, tweets):
    sequences = tokenizer.texts_to_sequences(tweets)
    padded_sequences = pad_sequences(sequences, truncating='post', maxlen=50, padding='post')
    return padded_sequences

index_to_classes = {0: 'sadness', 1: 'love', 2: 'joy', 3: 'anger', 4: 'fear', 5: 'surprise'}



model_emotions = tf.keras.models.load_model('emotion_model')



with open('tokenizer.pickle', 'rb') as handle:
    tokenizer = pickle.load(handle)


