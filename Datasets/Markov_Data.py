
from Models.model_sources.chain_source import ChainGenerator
from string import ascii_letters
from sklearn.model_selection import train_test_split

from Models.model_sources.path_encoder import PathEncoder
from datetime import datetime

class HMM_Data(object):

    @staticmethod
    def gen_data(state_size, order, size, verbose=False):
        cg = ChainGenerator(tuple(ascii_letters[:state_size]), order=order, min_len= order, max_len= order)
        print(cg.transition_matrix)
        x, y = cg.generate_data(size, random_state=datetime.now().second)
        pe = PathEncoder(order)
        pe.fit(x, y)

        x_tr3, y_tr3 = pe.transform(x, y)
        return train_test_split(x_tr3, y_tr3)
