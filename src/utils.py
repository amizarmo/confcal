import numpy as np
import pandas as pd
import re

AA_alphabet = 'BCDSQKIPTFNGHLRWAVEYMXZU-.*'
DNA_alphabet = 'ATCG-'
RNA_alphabet = 'UTCG-'


def remove_lower(text): return re.sub('[a-z]', '', text)


def chopping(data, lim):
    return [seq for seq in data if len(seq) <= lim]


def padding(data, lim, begin_token='', end_token='-',):
    padded = []
    for seq in data:
        temp = begin_token + seq + end_token * (lim - len(seq))
        padded.append(temp)
    return padded


def onehot_encoding(data, alphabet):
    aa2hot = {}
    for i, aa in enumerate(alphabet):
        v = [0 for j in alphabet]
        v[i] = 1
        aa2hot[aa] = v
        onehot_encoded = []
    for seq in data:
        temp = []
        for aa in seq:
            temp.append(aa2hot[aa])
        onehot_encoded.append(temp)
    return onehot_encoded


def onehot_decoding(data, alphabet):
    onehot_decoded = []
    for array in data:
        temp = ''
        for i, seq in enumerate(array):
            temp += alphabet[seq.index(max(seq))]
        onehot_decoded.append(temp)
    return onehot_decoded


def data_prep(data, x, y, lim, alphabet, flat):
    sequence_list = data[x].str.upper().tolist()
    sequence_list = chopping(sequence_list, lim=lim)
    sequence_list = padding(sequence_list, lim=lim)
    sequence_list = onehot_encoding(sequence_list, alphabet=alphabet)
    x_train = np.array(sequence_list, dtype=float)
    if flat:
        x_train = x_train.reshape(len(sequence_list), lim*len(alphabet))
    y_train = np.array(data[y], dtype=float)
    return x_train, y_train
