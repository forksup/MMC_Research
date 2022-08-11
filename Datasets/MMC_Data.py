import numpy as np
from Models.model_sources.chain_source import ChainGenerator
import random
from string import ascii_letters
import time
from numpy.random import choice
from Models import MMC
from sklearn.model_selection import train_test_split
from datetime import datetime

class MMC_data(object):

    @staticmethod
    def gen_data(state_size=4, order=1, size=50000, verbose=True):

        cg = ChainGenerator(list(ascii_letters[:state_size]),
                            order=1,
                            min_len= order,
                            max_len= order,
                            random_state=int(datetime.utcnow().timestamp()))

        labels = {v: k for k, v in cg._label_dict.items()}
        state_num = list(labels.values())

        probs = [(max(row), key) for key, row in enumerate(cg.transition_matrix)]
        probs.sort(reverse=True)
        SGO = [p[1] for p in probs]


        if verbose:
            print(cg.transition_matrix)
            print(SGO)

        rng = np.random.default_rng(int(time.time()))
        data = rng.choice(state_size, order)
        while(len(data) < size):
            data = np.append(data, rng.choice(state_num, 1, p=cg.transition_matrix[MMC.MMC.find_high(list(data[-order:]), SGO)])[0])

        data = list(map(int, data))
        x = [data[i:i + order] for i in range(len(data) - order)]
        y = [data[i] for i in range(order, len(data))]

        #preds = list(map(np.argmax,[cg.transition_matrix[MMC.MMC.find_high(x_, SGO)] for x_ in x]))
        #print("Maximum Accuracy:")
        #print(sum([1 for i in range(len(y)) if preds[i] == y[i]])/len(y))
        return train_test_split(np.asarray(x), np.asarray(y))

