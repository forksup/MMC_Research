import math
from collections import defaultdict
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
    def gen_data(states, order, size, verbose=False, four_blocks=False, drop_arms=False):

        if four_blocks:
            dataset = "Datasets/data_files/4blocks-10goals"
        else:
            dataset = "Datasets/data_files/6blocks"
        dataset = "Datasets/data_files/markovtraining_blocksworld.txt"

        data = pd.read_csv(dataset)

        if True:
            columns_to_drop = []
            for key in data.keys():
                # attempting to drop holding, arms, and action to make domain less deterministic
                if "holding" in key or "arm" in key:
                    columns_to_drop.append(key)
                    pass
            data.drop(columns_to_drop, axis=1, inplace=True)

        state_set = set()
        prev_index = 0
        episodes = []
        for i in data.loc[data['action'] == "end"].index:
            episodes.append(data.iloc[prev_index:i])
            prev_index = i+1


        def combine_two_pds(pd1,pd2):
                prob_to_continue = .7

                start = randint(0,1)
                indexleft = 0
                indexright = 0

                pd2 = pd2.rename(columns={key: key+"1" for key in pd2.keys()})
                pds = [pd1,pd2]
                left = []
                right = []

                def check_max(listobj,index):
                    return increment_index(listobj, index) == index

                def increment_index(listobj, index):
                    if len(listobj)-1 == index:
                        return index
                    return index + 1

                while indexleft < len(pd1) or indexright < len(pd2):
                    left.append(indexleft)
                    right.append(indexright)

                    if check_max(pd2, indexright) and check_max(pd1, indexleft):
                        break;
                    if check_max(pd2, indexright):
                        start = 0
                    elif check_max(pd1, indexleft):
                        start = 1
                    elif random.random() > prob_to_continue:
                        start = 1 - start

                    if start:
                        indexright += 1
                    else:
                        indexleft += 1

                combined_df = pd.concat([pd1.iloc[left].reset_index(), pd2.iloc[right].reset_index()], axis=1)
                columns = ["action", "action1"]
                landr = [left, right]
                for i in range(len(left)-1):

                    for l in range(len(columns)):

                        if landr[l][i] == landr[l][i+1]:
                            combined_df.at[i,columns[l]] = np.NaN
                return combined_df

        collect = []
        for i in range(len(data)-len(data)//2):
            collect.append(combine_two_pds(episodes[randint(0,len(episodes)-1)], episodes[randint(0,len(episodes)-1)]))

        data = pd.concat(collect)

        # Limit dataset to certain goal states
        end_states = defaultdict(list)
        start = 0
        b = 10
        transient_states = 2
        columns_to_add = ["on_b", "on-table_b", "clear_b"]

        """
        for i in range(transient_states):
            for c in columns_to_add:
                if "table" in c:
                    data[c+str(b+i)] = "True"
                else:
                    data[c+str(b+i)] = "null"
        stacked = False
        columns = {}
        """
        def set_item(data, dictionary):
            for key in dictionary:
                dictionary[key] = data[key]


        for key, row in data.iterrows():
            """
                    for i in data.loc[data['action'] == "end"].index:
            prev_index = i+1
            episodes.append(data.iloc[prev_index:i])"""
            if random.random() > .7:
                """
                if not stacked:
                    block_to_stack = randint(0, transient_states-1)
                    base_block = 0
                    if block_to_stack == 0:
                        base_block = 1
                    stacked = True

                    columns["on_b"+str(b+base_block)] = "b"+str(b+block_to_stack)
                    columns["clear_b"+str(b+base_block)] = "False"
                    columns["on-table_b"+str(b+block_to_stack)] = "False"
                else:
                    stacked = False
                    columns = {}
                """

            if row['on_b2'] == "end":
                end_states[tuple(data.iloc[key-1])].append(data.iloc[start:key+1])
                start = key+1
            else:
                ...
                #set_item(row, columns)
        

        sizes = []
        keys = []
        for key, d in end_states.items():
            sizes.append(len(d))
            keys.append(key)

        # limit the dataset to 10 random ending goals
        """
        top_30 = random.choices(list(range(len(keys))),k=15)

        data_states = [[]]
        all_data = []

        for t in top_30:
            all_data.extend(end_states[keys[t]])
        data = data
        """
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

        for i in range(len(y)):
            if random.random() > .8:
                y[i] = randint(0,max(y))
        if size > len(x):
            print("warning dataset size is greater than available blocksworld data")
        return train_test_split(np.asarray(x[:size]), np.asarray(y[:size])), len(state_keys)