import math
from collections import defaultdict
from random import randint
import pandas as pd

import random
import numpy as np
from sklearn.model_selection import train_test_split

# import necessary libraries
import pandas as pd
import os
import glob


class watchandhelp(object):
    state_keys = {}

    def gen_data(
        self, states, order, size, verbose=False, four_blocks=False, drop_arms=False
    ):
        # use glob to get all the csv files
        # in the folder
        path = "Datasets/data_files/new_watch_and_help_.8/"
        txt_files = glob.glob(os.path.join(path, "*.txt"))
        drop_consecutive_actions = False
        dataframes = []
        # loop over the list of csv files
        for f in txt_files:
            # read the csv file
            df = pd.read_csv(f, skipinitialspace=True)
            rows_to_drop = []
            for key, row in df.iterrows():
                if (
                    key != len(df) - 1
                    and row["Act_A"] == df.iloc[key + 1]["Act_A"]
                    and row["Act_B"] == df.iloc[key + 1]["Act_B"]
                ):
                    rows_to_drop.append(key)
            df = df.drop(labels=rows_to_drop, axis=0)
            df = df.reset_index(drop=True)
            dataframes.append(df)
        import random

        state_keys = {}
        episodes = [[]]
        count = 0

        for df in dataframes:
            for _, row in df.iterrows():
                converted = tuple(row)
                if not converted in state_keys:
                    state_keys[converted] = count
                    count += 1
                episodes[-1].append(state_keys[converted])
            episodes.append([])
        del episodes[-1]

        x = []
        y = []

        for e in episodes:
            x.extend([e[i : i + order] for i in range(len(e) - order)])
            y.extend([e[i] for i in range(order, len(e))])

        noise = 0
        print(f"Noise: {noise*100}%")
        for i in range(len(y)):
            x[i][-1] = y[randint(0, len(y) - 1)]
            if random.random() < noise:
                y[i] = y[randint(0, len(y) - 1)]

        return train_test_split(np.array(x), np.array(y))
