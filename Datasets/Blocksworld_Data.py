import math
from random import randint
from datetime import datetime

from fastparquet import ParquetFile
import pandas as pd
import random
import numpy as np
from numpy.random import choice
from Models import MMC
from sklearn.model_selection import train_test_split


class blocks(object):

    @staticmethod
    def gen_data(states, order, size, verbose=False, four_blocks=False, drop_arms=True):
        if four_blocks:
            dataset = "Datasets/data_files/markovtraining_blocksworld.txt"
        else:
            dataset = "Datasets/data_files/6blocks"
        data = pd.read_csv(dataset)
        if drop_arms:
            columns_to_drop = []
            for key in data.keys():
                if "holding" in key or "arm" in key or "act" in key:
                    columns_to_drop.append(key)
                    pass
            data.drop(columns_to_drop, axis=1, inplace=True)

        state_set = set()
        state_keys = {}
        for key, row in data.iterrows():
            if not row['on_b2'] == "end":
                state_set.add(tuple(row))
        count = 0
        for s in state_set:
            state_keys[s] = count
            count += 1
        data_states = [[]]
        for key, row in data.iterrows():
            if not row['on_b2'] == "end":
                data_states[-1].append(state_keys[tuple(row)])
            else:
                data_states.append([])

        x = []
        y = []
        for e in data_states:
            e = list(e)
            x.extend([e[i:i+order] for i in range(len(e)-order)])
            y.extend([e[i] for i in range(order, len(e))])
        if size > len(x):
            print("warning dataset size is greater than available blocksworld data")
        return train_test_split(np.asarray(x[:size]), np.asarray(y[:size])), len(state_keys)