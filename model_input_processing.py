#!/usr/bin/env python
# -*- coding: utf-8 -*-

import logging
import pandas as pd

def hash_map_generator(w2v_file):
    raw_txt = open(w2v_file)
    raw_txt = raw_txt.read()
    raw_txt = raw_txt.split('\n')
    raw_txt = [x.split() for x in raw_txt]
    raw_txt.pop()
    name = []
    vectors =[]
    for element in raw_txt:
        name.append(element[0])
        vector = element[1:]
        vector = [float(x) for x in vector]
        vectors.append(vector)
    hash_map = dict()
    for index in range(len(name)):
        hash_map.update({name[index]:vectors[index]})
    return hash_map

def detect_input_length(twitter_file):
    raw_data = pd.read_csv(twitter_file)
    return max(len(l) for l in raw_data['text'].str.split())

def input_generator(twitter_file, w2v_file):
    vector_length = 50
    raw_data = pd.read_csv(twitter_file)
    raw_data_split = raw_data['text'].str.split()
    max_len = detect_input_length(twitter_file)
    padding = [0.0] * vector_length
    input_data = []
    hash_map = hash_map_generator(w2v_file)
    for twitt_idx in range(len(raw_data)):
        input_data.append([])
        for word_idx in range(max_len):
            if word_idx < len(raw_data_split[twitt_idx]):
                if raw_data_split.iloc[twitt_idx][word_idx] in hash_map:
                    input_data[twitt_idx].append(hash_map[raw_data_split.iloc[twitt_idx][word_idx]])
                else:
                    input_data[twitt_idx].append(hash_map['<unk>'])
            else:
                input_data[twitt_idx].append(padding)
    return input_data

def dataset_generator(twitter_file, w2v_file, training_size_ratio, label):
    raw_data = pd.read_csv(twitter_file)
    input_data = input_generator(twitter_file, w2v_file)
    input_label = list(raw_data[label])
    training_size = len(input_data) * training_size_ratio
    training_size = int(training_size)
    training_data = input_data[0:training_size]
    training_label = input_label[0:training_size]
    testing_data = input_data[training_size:len(input_data)]
    testing_label = input_label[training_size:len(input_data)]
    return training_data, training_label, testing_data, testing_label
