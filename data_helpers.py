#-*- coding:utf8 -*-
import numpy as np
import re
import itertools
from collections import Counter
from collections import defaultdict
import sys
import pickle
'''
def clean_str(string):
    """
    Tokenization/string cleaning for all datasets except for SST.
    Original taken from https://github.com/yoonkim/CNN_sentence/blob/master/process_data.py
    """
    string = re.sub(r"[^A-Za-z0-9(),!?\'\`]", " ", string)
    string = re.sub(r"\'s", " \'s", string)
    string = re.sub(r"\'ve", " \'ve", string)
    string = re.sub(r"n\'t", " n\'t", string)
    string = re.sub(r"\'re", " \'re", string)
    string = re.sub(r"\'d", " \'d", string)
    string = re.sub(r"\'ll", " \'ll", string)
    string = re.sub(r",", " , ", string)
    string = re.sub(r"!", " ! ", string)
    string = re.sub(r"\(", " \( ", string)
    string = re.sub(r"\)", " \) ", string)
    string = re.sub(r"\?", " \? ", string)
    string = re.sub(r"\s{2,}", " ", string)
    return string.strip().lower()
'''


def load_data_and_labels(home_data,work_data,school_data,restaurant_data,shopping_data,cinema_data,sports_data,travel_data):
    """
    Loads MR polarity data from files, splits the data into words and generates labels.
    Returns split sentences and labels.
    """
    # Load data from files
    home_examples = list(open(home_data, "r",encoding='UTF-8').readlines())
    home_text = [s.strip().split('  ')[2].split(':')[1].strip() for s in home_examples]
    home_text=np.array(home_text)
    home_image=[s.strip().split('  ')[4].split(':')[1].strip().split(' ') for s in home_examples]
    home_image=np.array(home_image)
    home_user=[s.strip().split('  ')[0].split(':')[1].strip().split() for s in home_examples]
    home_user=np.array(home_user)   

    work_examples = list(open(work_data, "r",encoding='UTF-8').readlines())
    work_text = [s.strip().split('  ')[2].split(':')[1].strip() for s in work_examples]
    work_text=np.array(work_text)
    work_image=[s.strip().split('  ')[4].split(':')[1].strip().split(' ') for s in work_examples]
    work_image=np.array(work_image)
    work_user=[s.strip().split('  ')[0].split(':')[1].strip().split() for s in work_examples]
    work_user=np.array(work_user)

    school_examples = list(open(school_data, "r",encoding='UTF-8').readlines())
    school_text = [s.strip().split('  ')[2].split(':')[1].strip() for s in school_examples]
    school_text=np.array(school_text)
    school_image=[s.strip().split('  ')[4].split(':')[1].strip().split(' ') for s in school_examples]
    school_image=np.array(school_image)
    school_user=[s.strip().split('  ')[0].split(':')[1].strip().split() for s in school_examples]
    school_user=np.array(school_user)

    restaurant_examples = list(open(restaurant_data, "r",encoding='UTF-8').readlines())
    restaurant_text = [s.strip().split('  ')[2].split(':')[1].strip() for s in restaurant_examples]
    restaurant_text=np.array(restaurant_text)
    restaurant_image=[s.strip().split('  ')[4].split(':')[1].strip().split(' ') for s in restaurant_examples]
    restaurant_image=np.array(restaurant_image)
    restaurant_user=[s.strip().split('  ')[0].split(':')[1].strip().split() for s in restaurant_examples]
    restaurant_user=np.array(restaurant_user)

    shopping_examples = list(open(shopping_data, "r",encoding='UTF-8').readlines())
    shopping_text = [s.strip().split('  ')[2].split(':')[1].strip() for s in shopping_examples]
    shopping_text=np.array(shopping_text)
    shopping_image = [s.strip().split('  ')[4].split(':')[1].strip().split(' ') for s in shopping_examples]
    shopping_image=np.array(shopping_image)
    shopping_user = [s.strip().split('  ')[0].split(':')[1].strip().split() for s in shopping_examples]
    shopping_user=np.array(shopping_user)

    cinema_examples = list(open(cinema_data, "r",encoding='UTF-8').readlines())
    cinema_text = [s.strip().split('  ')[2].split(':')[1].strip() for s in cinema_examples]
    cinema_text=np.array(cinema_text)
    cinema_image = [s.strip().split('  ')[4].split(':')[1].strip().split(' ') for s in cinema_examples]
    cinema_image=np.array(cinema_image)
    cinema_user = [s.strip().split('  ')[0].split(':')[1].strip().split() for s in cinema_examples]
    cinema_user=np.array(cinema_user)

    sports_examples = list(open(sports_data, "r",encoding='UTF-8').readlines())
    sports_text = [s.strip().split('  ')[2].split(':')[1].strip() for s in sports_examples]
    sports_text=np.array(sports_text)
    sports_image = [s.strip().split('  ')[4].split(':')[1].strip().split(' ') for s in sports_examples]
    sports_image=np.array(sports_image)
    sports_user = [s.strip().split('  ')[0].split(':')[1].strip().split() for s in sports_examples]
    sports_user=np.array(sports_user)

    travel_examples = list(open(travel_data, "r",encoding='UTF-8').readlines())
    travel_text = [s.strip().split('  ')[2].split(':')[1].strip() for s in travel_examples]
    travel_text=np.array(travel_text)
    travel_image = [s.strip().split('  ')[4].split(':')[1].strip().split(' ') for s in travel_examples]
    travel_image=np.array(travel_image)
    travel_user = [s.strip().split('  ')[0].split(':')[1].strip().split() for s in travel_examples]
    travel_user=np.array(travel_user)

    # Split by words
    x_text =np.concatenate((home_text,work_text,school_text,restaurant_text,shopping_text, cinema_text, sports_text,travel_text),0)
    x_image=np.concatenate((home_image,work_image,school_image,restaurant_image,shopping_image, cinema_image, sports_image,travel_image),0)
    x_user=np.concatenate((home_user,work_user,school_user,restaurant_user,shopping_user, cinema_user, sports_user,travel_user),0)
    # Generate labels
    home_labels = [[1,0,0,0,0,0,0,0] for _ in home_examples]
    work_labels = [[0,1,0,0,0,0,0,0] for _ in work_examples]
    school_labels = [[0,0,1,0,0,0,0,0] for _ in school_examples]
    restaurant_labels = [[0,0,0,1,0,0,0,0] for _ in restaurant_examples]
    shopping_labels = [[0,0,0,0,1,0,0,0] for _ in shopping_examples]
    cinema_labels = [[0,0,0,0,0,1,0,0] for _ in cinema_examples]  
    sports_labels = [[0,0,0,0,0,0,1,0] for _ in sports_examples]
    travel_labels = [[0,0,0,0,0,0,0,1] for _ in travel_examples]
    y = np.concatenate([home_labels,work_labels,school_labels,restaurant_labels,shopping_labels,cinema_labels,sports_labels,travel_labels], 0)
    return [x_text,x_image,x_user,y]


def loadDict(dict_file):
    output_dict = None
    with open(dict_file, 'rb') as f:
        output_dict = pickle.load(f)
    return output_dict


def saveDict(input_dict, output_file):
    with open(output_file, 'wb') as f:
        pickle.dump(input_dict, f) 


def batch_iter(data, batch_size, num_epochs, shuffle=True):
    """
    Generates a batch iterator for a dataset.
    """
    data = np.array(data)
    data_size = len(data)
    num_batches_per_epoch = int((len(data)-1)/batch_size) + 1
    for epoch in range(num_epochs):
        # Shuffle the data at each epoch
        if shuffle:
            shuffle_indices = np.random.permutation(np.arange(data_size))
            shuffled_data = data[shuffle_indices]
        else:
            shuffled_data = data
        for batch_num in range(num_batches_per_epoch):
            start_index = batch_num * batch_size
            end_index = min((batch_num + 1) * batch_size, data_size)
            yield shuffled_data[start_index:end_index]
