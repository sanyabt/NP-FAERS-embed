#!/usr/bin/env python

import tensorflow as tf
import string
import random
import re

def add_noise(w, percent):
    ''' edit, del, add'''
    positions = random.choices(range(len(w)), k=int(percent*len(w)))
    for p in positions:
        r = random.random()
        if r <= 1/3: # edit
            w = w[:p] + random.choice(string.ascii_uppercase) + w[p+1:]
        elif r<= 2/3: # delete
            w = w[:p] + w[p+1:]
        elif r<=1: # add
            w = w[:p] + random.choice(string.ascii_uppercase) + w[p:]
    return w

def clean(text):
    #remove all non-ascii, special characters and keep alphabets and space only. Can also use isalpha()
    #convert to uppercase , remove extra spaces 
    regex = re.compile('[^a-zA-Z ]')
    r = regex.sub('', text)
    result = re.sub(' +', ' ', r)
    result = result.strip()
    return result.upper()

def clean_dataset(data):
    x = []
    y = []
    for i in range(data.shape[0]):
        w = clean(data.FAERS_drug_match.iloc[i])
        v = clean(data.lookup_value.iloc[i])
        x.append(w)
        y.append(v)
    return x,y

def clean_dataset_1d(df):
    return [clean(x) for x in df.FAERS_drug_match]

def encode_dataset(x,y=None):
    encode_dict = {l:i+1 for i,l in enumerate(string.ascii_uppercase + " ")}
    Xtrain = [[encode_dict[m] for m in n] for n in x]
    if y:
        Ytrain = [[encode_dict[m] for m in n] for n in y]
        return Xtrain, Ytrain
    return Xtrain

def clean_encode_padding(q, maxlen):
    q = clean(q)
    encode_dict = {l:i+1 for i,l in enumerate(string.ascii_uppercase + " ")}
    return tf.keras.preprocessing.sequence.pad_sequences(
        [encode_dict[m] for m in q] , padding="post", maxlen=maxlen)

def padding_dataset(X,Y=None,maxlen=400):
    padded_x = tf.keras.preprocessing.sequence.pad_sequences(
        X, padding="post", maxlen=maxlen)
    if Y:
        padded_y = tf.keras.preprocessing.sequence.pad_sequences(
            Y, padding="post", maxlen=maxlen)
    
        return padded_x, padded_y
    return padded_x

def cosine_distance(vects):
    x, y = vects
    return 1-tf.reduce_sum(tf.multiply(x,y),axis=1, keepdims=True)/(tf.norm(x,axis=1,keepdims=True)*tf.norm(y,axis=1,keepdims=True))

def loss(margin=1):
    def contrastive_loss(y_true, y_pred):
        square_pred = tf.math.square(y_pred)
        margin_square = tf.math.square(tf.math.maximum(margin - (y_pred), 0))
        return tf.math.reduce_mean(
            (1 - y_true) * square_pred + (y_true) * margin_square
        )

    return contrastive_loss

def plt_metric(history, metric, title, has_valid=True):
    """Plots the given 'metric' from 'history'.

    Arguments:
        history: history attribute of History object returned from Model.fit.
        metric: Metric to plot, a string value present as key in 'history'.
        title: A string to be used as title of plot.
        has_valid: Boolean, true if valid data was passed to Model.fit else false.

    Returns:
        None.
    """
    plt.plot(history[metric])
    if has_valid:
        plt.plot(history["val_" + metric])
        plt.legend(["train", "validation"], loc="upper left")
    plt.title(title)
    plt.ylabel(metric)
    plt.xlabel("epoch")
    plt.show()
    

