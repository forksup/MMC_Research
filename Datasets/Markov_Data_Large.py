import random

from Models.model_sources.chain_source import ChainGenerator
from string import ascii_letters
from sklearn.model_selection import train_test_split
from random import randint, uniform
from Models.model_sources.path_encoder import PathEncoder
from datetime import datetime

class HMM_Decisive(object):

    @staticmethod
    def gen_data(state_size, order, size, verbose=False):
        cg = ChainGenerator(tuple(ascii_letters[:state_size]), order=order, min_len= order, max_len= order)
        for s,v in cg._label_dict.items():
            state_to_change = randint(0,state_size-1)
            min_rand = 1/state_size + .1
            new_prob = uniform(min_rand, .9)
            cg.transition_matrix[s][state_to_change] = new_prob
            rem = 1 - new_prob
            for st in range(state_size):
                if st == state_to_change:
                    continue
                prob = uniform(0, rem)
                cg.transition_matrix[s][st] = prob
                rem -= prob

        print(cg.transition_matrix)
        x, y = cg.generate_data(size, random_state=datetime.now().second)
        pe = PathEncoder(order)
        pe.fit(x, y)

        x_tr3, y_tr3 = pe.transform(x, y)
        return train_test_split(x_tr3, y_tr3)
