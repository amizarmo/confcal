import numpy as np
from sklearn.linear_model import LogisticRegression


def get_LR():
    model = LogisticRegression(
        solver='liblinear', random_state=3)
    return model


def get_RNN():
    return


def get_CNN():
    return
