import numpy as np
from sklearn.model_selection import train_test_split
import pandas as pd
class fruit_domain(object):

    @staticmethod
    def gen_data(states, order, size, verbose=False):
        df = pd.read_pickle("/mnt/watchandhelp/PycharmProjects/thesis_test/Datasets/data_files/fruit.pkl")
        df = df.head(size)
        x = [df[i:i+order].to_numpy().flatten() for i in range(len(df)-order)]
        y = [df.iloc[i]['state'] for i in range(order, len(df))]
        return train_test_split(np.asarray(x), np.asarray(y))

