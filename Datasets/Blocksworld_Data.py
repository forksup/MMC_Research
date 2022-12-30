import math
from collections import defaultdict
from random import randint
import pandas as pd

import random
import numpy as np
from sklearn.model_selection import train_test_split


class blocks(object):
    state_keys = {}

    @staticmethod
    def gen_data(states, order, size, verbose=False, four_blocks=False, drop_arms=False):
        load_data = False

        if four_blocks:
            dataset = "Datasets/data_files/4blocks-10goals"
        else:
            dataset = "Datasets/data_files/6blocks"
        dataset =  "Datasets/data_files/6blocks"

        if load_data:
            data = pd.read_csv(f"{dataset}.csv")
        else:
            block_count = 4
            print(f"Dataset: {dataset}")
            data = pd.read_csv(dataset)

            if True:
                columns_to_drop = []
                for key in data.keys():
                    # attempting to drop holding, arms, and action to make domain less deterministic
                    if "holding" in key or "arm" in key:
                        columns_to_drop.append(key)
                        pass
                data.drop(columns_to_drop, axis=1, inplace=True)

            prev_index = 0
            episodes = []
            for i in data.loc[data['clear_b1'] == "end"].index:
                # only add episodes with length greater than 1
                if i - prev_index > 1:
                    episodes.append(data.iloc[prev_index:i])
                prev_index = i + 1

            def combine_two_pds(pd1, pd2):
                prob_to_continue = .7

                start = randint(0, 1)
                indexleft = 0
                indexright = 0
                # rename blocks on right side
                rename_dict = {}
                for key in pd2.keys():
                    if key != "action":
                        rename_dict[key] = key.split("_")[0] + f"_b{int(key.split('_')[1][1]) + block_count}"

                pd2 = pd2.rename(columns=rename_dict)

                rename_action_dict = {}
                for act in pd2['action'].unique():
                    new_act = act
                    if isinstance(act, str) or not math.isnan(act):
                        for s in act.split("_")[1:]:
                            new_act = new_act.replace(s[1], str(int(s[1]) + block_count))
                    rename_action_dict[act] = new_act
                pd2['action'] = pd2['action'].replace(rename_action_dict)
                left = []
                right = []

                def check_max(listobj, index):
                    return increment_index(listobj, index) == index

                def increment_index(listobj, index):
                    if len(listobj) - 1 == index:
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
                # change this so there is one action column, change the blocks name
                # blocks 1-4 on left and blocks 5-8 on right
                # combine them to one action column
                # let's do one action
                combined_df = pd.concat([pd1.iloc[left].reset_index(), pd2.iloc[right].reset_index()], axis=1)
                combined_df = combined_df.reset_index()
                columns = ["action"]
                for i in range(len(combined_df)):
                    # left changes
                    # right changes
                    if right[i] == max(right) or right[i] == right[i + 1]:
                        combined_df.at[i, 'action'] = pd1.iloc[left[i]]['action']
                    else:
                        combined_df.at[i, 'action'] = pd2.iloc[right[i]]['action']
                combined_df = combined_df.drop('level_0', axis=1)
                return combined_df.drop('index', axis=1)

            print("Generating Data Randomly")
            collect = []
            for _ in range(size):
                episode1 = randint(0, len(episodes) - 1)
                episode2 = randint(0, len(episodes) - 1)
                for _ in range(40):
                    collect.append(combine_two_pds(episodes[episode1], episodes[episode2]))
            print(f"Dataset Size: {len(collect)}")
            # here we need to enter end states
            data = pd.concat(collect)
            data.to_csv(f"{dataset}.csv")

        # data = data.iloc[0:10000]
        # Limit dataset to certain goal states
        try:
            data = data.drop('Unnamed: 0', axis=1)
        except:
            ...
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

        """
        for key, row in data.iterrows():

                    for i in data.loc[data['action'] == "end"].index:
            prev_index = i+1
            episodes.append(data.iloc[prev_index:i])
            if random.random() > .7:

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

            if row['on_b2'] == "end":
                end_states[tuple(data.iloc[key - 1])].append(data.iloc[start:key + 1])
                start = key + 1
            else:
                ...
                # set_item(row, columns)
        sizes = []
        keys = []
        for key, d in end_states.items():
            sizes.append(len(d))
            keys.append(key)
        """
        data_states = [[]]
        count = 0
        for key, row in data.iterrows():
            if not row['on_b2'] == "end":
                conv = tuple(row)
                if conv not in state_keys:
                    state_keys[conv] = count
                    count += 1

                data_states[-1].append(state_keys[conv])
            else:
                data_states.append([])
        print(f"State Size: {count}")
        x = []
        y = []
        for e in data_states:
            e = list(e)
            x.extend([e[i:i + order] for i in range(len(e) - order)])
            y.extend([e[i] for i in range(order, len(e))])

        x = np.asarray(x[:size])
        y = np.asarray(y[:size])

        # test
        noise = .20
        print(f"Noise: {noise * 100}%")
        for i in range(len(y)):
            x[i][-1] = y[randint(0, len(y) - 1)]
            if random.random() < noise:
                y[i] = y[randint(0, len(y) - 1)]

        if size > len(x):
            print("warning dataset size is greater than available blocksworld data")
        return train_test_split(x, y), len(self.state_keys)