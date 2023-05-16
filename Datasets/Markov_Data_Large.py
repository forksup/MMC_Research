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
        cg = ChainGenerator(
            tuple(ascii_letters[:state_size]), order=order, min_len=order, max_len=order
        )
        for s, v in cg._label_dict.items():
            state_to_change = randint(0, state_size - 1)
            min_rand = 1 / state_size + 0.1
            new_prob = uniform(min_rand, 0.9)
            cg.transition_matrix[s][state_to_change] = new_prob
            rem = 1 - new_prob
            for st in range(state_size):
                if st == state_to_change:
                    continue
                if st == state_size - 1:
                    cg.transition_matrix[s][st] = rem
                else:
                    prob = uniform(0, rem)
                    cg.transition_matrix[s][st] = prob
                    rem -= prob

        for s in range(len(cg.transition_matrix)):
            total = sum(cg.transition_matrix[s])
            if total != 1:
                for ss in range(len(cg.transition_matrix[s])):
                    cg.transition_matrix[s][ss] /= total
        if verbose:
            print(cg.transition_matrix)

        # now normalize in case they do not sum to 1

        x, y = cg.generate_data(size, random_state=datetime.now().second)
        pe = PathEncoder(order)
        pe.fit(x, y)

        x_tr3, y_tr3 = pe.transform(x, y)
        return train_test_split(x_tr3, y_tr3)
