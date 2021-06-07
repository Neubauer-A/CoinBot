#!/usr/local/bin/python

import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'
import argparse
import pandas as pd
import tensorflow as tf
import tensorflow_hub as hub
import tensorflow_text as text
from official.nlp import optimization 

def build_classifier_model(
    bert_model='https://tfhub.dev/google/experts/bert/wiki_books/sst2/2',
    preprocess_model='https://tfhub.dev/tensorflow/bert_en_uncased_preprocess/3'
    ):
    preprocess = hub.load(preprocess_model)
    bert = hub.load(bert_model)

    text_input = tf.keras.layers.Input(shape=(), dtype=tf.string)
    preprocessing_layer = hub.KerasLayer(preprocess, name='preprocessing')
    encoder_inputs = preprocessing_layer(text_input)
    encoder = hub.KerasLayer(bert, trainable=True, name='BERT_encoder')
    outputs = encoder(encoder_inputs)
    net = outputs['pooled_output']
    net = tf.keras.layers.Dropout(0.1)(net)
    net = tf.keras.layers.Dense(3, activation='softmax', name='classifier')(net)
    return tf.keras.Model(text_input, net)

def get_data(train_path, test_path, features, labels):
    df = pd.read_csv(train_path)
    x_train = df[features].values
    y_train = df[labels].values
    y_train = tf.keras.utils.to_categorical(y_train, num_classes=3)

    df = pd.read_csv(test_path)
    x_test = df[features].values
    y_test = df[labels].values
    y_test = tf.keras.utils.to_categorical(y_test, num_classes=3)
    return x_train, y_train, x_test, y_test

def finetune(model, x_train, y_train, x_test, y_test, epochs, batch_size):
    steps_per_epoch = int(len(x_train)/batch_size)
    num_train_steps = steps_per_epoch * epochs
    num_warmup_steps = int(0.1*num_train_steps)

    init_lr = 3e-5
    optimizer = optimization.create_optimizer(init_lr=init_lr,
                                              num_train_steps=num_train_steps,
                                              num_warmup_steps=num_warmup_steps,
                                              optimizer_type='adamw')
    model.compile(optimizer=optimizer,
                  loss='categorical_crossentropy',
                  metrics='accuracy')
    model.fit(x_train, y_train, epochs=epochs, batch_size=batch_size, 
              shuffle=True, validation_data=(x_test, y_test), verbose=0)
    
def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--train-path', type=str, default='CoinBot/data/datasets/combined_train.csv')
    parser.add_argument('--test-path', type=str, default='CoinBot/data/datasets/combined_valid.csv')
    parser.add_argument('--features', type=str, default='text')
    parser.add_argument('--labels', type=str, default='label')
    parser.add_argument('--epochs', type=int, default=4)
    parser.add_argument('--batch-size', type=int, default=32)
    parser.add_argument('--save-path', type=str, default='CoinBot/sentiment/btcBERT')
    args = parser.parse_args()

    model = build_classifier_model()
    x_train, y_train, x_test, y_test = get_data(args.train_path, args.test_path, 
                                                args.features, args.labels)
    finetune(model, x_train, y_train, x_test, y_test, args.epochs, args.batch_size)
    model.save(args.save_path, include_optimizer=False)

if __name__ == '__main__':
    main()