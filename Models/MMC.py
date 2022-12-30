from copy import deepcopy
import itertools
from collections import defaultdict, Counter
import random
from enum import Enum
import Models.MAX
import numpy as np
import math
from numpy.random import choice


class sgo_types(Enum):
    greedy = 1
    hillclimb = 2
    full = 3
    geometric_mean = 4


class MMC(object):

    def __init__(self, state_size, order, sgo_method: sgo_types = sgo_types.greedy, verbose=False):
        self.sgom = sgo_method
        self.state_size = state_size
        self.states = [i for i in range(state_size)]
        self.cpt = None
        self.SGO = None
        self.name = "MMC"
        self.verbose = verbose
        self.index_dict = {}
        self.misalignment = True

    @staticmethod
    def find_high(lag, index_dict):
        return min(lag, key=lambda x: index_dict[x])
        # return min(lag, key=lambda x: self.index_dict[x])

    def geometric_mean(self):
        # use greedy for hillclimb
        # increase domain size too
        #
        sgo = []
        states_left_to_check = set(self.states)
        while len(states_left_to_check) > 1:
            states = []
            for s in states_left_to_check:
                self.index_dict = defaultdict(lambda: float('inf'))
                self.index_dict[s] = 0
                prob_res = list(self.calculate_probabilities([s])[0][s].values())
                c = sum(prob_res)
                p_values = {i: prob_res[i] / c for i in range(len(prob_res))}
                divident = np.sum([(math.log(p_values[i]) * n) for i, n in enumerate(prob_res)])
                calc = (divident * -1) ** (1 / c)
                states.append((calc, s))
            states.sort(reverse=True)
            sgo.append(states[0][1])
            states_left_to_check -= {sgo[-1]}
        sgo.append(list(states_left_to_check)[0])
        return sgo

    def find_SGO_hill_climb(self):
        # sort by data generation probability selecting one state at a time
        # find data generation probabilities with SGO length = 1
        """
        left_to_check = set(self.states)
        SGO = []

        probs = []
        states = []
        for s in left_to_check:
            n = self.calculate_probabilities([s])[s]
            x = 0
            for val in n.values():
                x *= val
            x^
            """
        SGO = self.geometric_mean()
        n, normalized = self.calculate_probabilities(SGO)

        prob = self.gen_prob(n, normalized)
        while True:
            for i in range(len(SGO)):
                for j in range(i, len(SGO)):
                    new_SGO = deepcopy(SGO)
                    new_SGO[i], new_SGO[j] = new_SGO[j], new_SGO[i]

                    n, normalized = self.calculate_probabilities(new_SGO)
                    new_prob = self.gen_prob(n, normalized)
                    if new_prob > prob:
                        prob = new_prob
                        SGO = new_SGO
                else:
                    continue
                break
            else:
                break

        return SGO

    def calculate_probabilities(self, sgo):
        n = defaultdict(lambda: defaultdict(float))

        for i, lag in enumerate(self.X_train):
            s = self.find_high(lag, self.index_dict)
            if s is not float('-inf'):
                n[s][self.y_train[i]] += 1
            else:
                pass
                # print("potential error")
        for s in self.states:
            if s not in n:
                n[s] = {s: 1 / self.state_size for s in self.states}
        return deepcopy(n), self.normalize_dct(n)

    def normalize_dct(self, n):
        for k1, val in n.items():
            tot = sum(val.values())
            for k2 in val:
                n[k1][k2] /= tot
        return n

    # this function creates a graph of each state in the SGO. misalignments are connected via edges
    # and states with no misalignmnts are independent nodes
    def combine_misalignments(self, l, sets):
        for i, s in enumerate(sets):
            if l.intersection(s):
                sets[i] = s.union(l)
                return
        else:
            sets.append(l)

    def gen_prob(self, n, probs):
        prod = 1
        for kn, d in n.items():
            for key, value in d.items():
                prod += n[kn][key] * math.log(probs[kn][key])
        return prod

    def gen_prob_dict(self, n, probs):
        result = defaultdict(dict)

        for kn, d in n.items():
            for key, value in d.items():
                result[kn][key] = n[kn][key] * math.log(1+probs[kn][key])
        return result

    @staticmethod
    def create_index_dict(sgo):
        index = defaultdict(lambda: float('inf'))
        for i, s in enumerate(sgo):
            index[s] = i
        return index

    def find_SGO_greedy(self):
        states_to_check = set(self.states) - {-1}
        SGO = []

        def func(p):
            if n[p].values():
                total = sum(n[p].values())
                a = {k: v / total for k, v in n[p].items()}
                return max(a.values())
            else:
                return 0

        index_dict = defaultdict(lambda: float('inf'))

        while len(states_to_check) > 1:

            n = defaultdict(lambda: defaultdict(float))

            for i, lag in enumerate(self.X_train):
                s = self.find_high(lag, index_dict)
                if s in states_to_check:
                    for st in lag:
                        n[st][self.y_train[i]] += 1
            # In the case all the data already has a high state
            if len(n) == 0:
                for s in states_to_check:
                    SGO.append(s)
                return SGO

            s = max(n, key=func)
            index_dict[s] = len(SGO)
            SGO.append(s)
            if s not in states_to_check:
                raise Exception(
                    "State does not exist in states to check. Potentially incorrect amount of states provided")
            states_to_check.remove(s)
        SGO.append(states_to_check.pop())
        return SGO

    def train(self, X_train, y_train):

        self.states = set().union(*[np.unique(X_train), np.unique(y_train)])
        self.X_train = X_train
        self.y_train = y_train
        SGOs = []
        if self.sgom == sgo_types.greedy:
            SGOs = [self.find_SGO_greedy()]
        elif self.sgom == sgo_types.hillclimb:
            SGOs = [self.find_SGO_hill_climb()]
        elif self.sgom == sgo_types.full:
            SGOs = list(itertools.permutations([s for s in range(self.state_size) if s != -1]))
            random.shuffle(SGOs)
        elif self.sgom == sgo_types.geometric_mean:
            SGOs = [self.geometric_mean()]
        sgo_results = []
        for SGO in SGOs:
            if len(SGOs) == 1:
                self.index_dict = self.create_index_dict(SGO)
                count_dict, n = self.calculate_probabilities(SGO)
                sgo_results.append((-1, SGO, deepcopy(n)))
            else:
                sgo_results.append((self.gen_prob(count_dict, n), SGO, deepcopy(n)))

            genprobs = self.gen_prob_dict(count_dict, n)
            # self.cpt = n
            # self.build_cpt()
            # print(self.cpt)

            max_probs = defaultdict(float)
            for k, val in genprobs.items():
                max_probs[k] = max(val.values())

            mis_alignment = []
            # change this so we grab sgo from new data generation problem
            # uncomment out this section 
            # Retrieve all combinations of any two states to check for misalignments
            # misalignment in java code
            # the remaining items
            sgo_from_data = sorted(max_probs, key=max_probs.get, reverse=True)

            i_fromdata = {s: sgo_from_data.index(s) for s in self.states}
            i_fromsgo = {s: SGO.index(s) for s in self.states}

            state_index = {s:i for s,i in enumerate(self.states)}


            set1 = set(sgo_from_data)
            mis_alignment1 = []

            if self.verbose:
                print(f"Combine misalignment: {self.misalignment}")
            if self.misalignment:
                for i in range(len(i_fromdata) - 1):
                    if SGO[i] != sgo_from_data[i]:
                        if i_fromsgo[SGO[i]] < i_fromdata[SGO[i]]:
                            self.combine_misalignments(set(SGO[i:i_fromdata[SGO[i]]+1]), mis_alignment1)

                        """
                        set_diff = set1 - set(SGO[i_fromsgo[sgo_from_data[i]] :])
                        if set_diff:
                            self.combine_misalignments(set(set_diff).union({sgo_from_data[i]}), mis_alignment)
                        """
                    set1.remove(sgo_from_data[i])

            # all of the states inbetween where its suppoed to be are misaligned
            if self.verbose:

                print(f"SGO: {SGO}")
                print(f"{mis_alignment}")


            for m in mis_alignment1:
                # combine the count of the maximum key for each probability value then divide that by the
                # total occurences for every state

                # max(count_dict[0], key=count_dict[0].get)
                # [max(count_dict[k], key=count_dict[k].get) for k in count_dict]
                # Gather all probability values for all misalligned states
                input_table = {s: count_dict[s] for s in m}
                new_table = {}


                # Combine all the cpt's
                res = dict(sum((Counter(dict(x)) for x in input_table.values()), Counter()))
                max_values = [max(input_table[key].values()) for key in input_table]

                max_prob = sum(max_values) / sum(res.values())

                for key in new_table:

                    # find the key of the max value
                    max_key = max(input_table[key], key=input_table[key].get)

                    rem = 1 - max_prob
                    tot = sum(new_table[key].values()) - new_table[key][max_key]
                    new_table[key][max_key] = max_prob
                    running_sum = max_prob

                    for i in range(self.state_size):
                        new_table[key][i] = input_table[key][i]

                    # loop through all the rest of the items in descending order of probability
                    for k2 in dict(sorted(new_table[key].items(), key=lambda item: item[1], reverse=True)):

                        if k2 == max_key:
                            continue

                        if new_table[key][k2] == 0:
                            print("setting zero")
                            # equally distribute the remainder across 0 values
                            lval = [k3 for k3, v2 in count_dict[key].items() if v2 == 0]
                            probtoset = rem / len(lval)
                            for ktset in lval:
                                new_table[key][ktset] = probtoset
                            break

                        nns = (new_table[key][k2] / tot) * rem
                        if nns > max_prob:
                            nns = min(rem, max_prob)

                            rem -= nns
                            rem = max(rem, 0)

                            running_sum += nns
                            new_table[key][k2] = nns
                            tot = sum(new_table[key].values()) - running_sum
                        else:
                            running_sum += nns
                            new_table[key][k2] = nns
                    if len(new_table) != self.state_size:
                        print("length mismatch")
                    if self.verbose:
                            print(sum(new_table[key].values()))
                for key in new_table:
                    n[key] = new_table

        if self.verbose:
            for key in n:
                print(sum(n[key].values()))

        if len(SGOs) > 1:
            sgo_results.sort(key=lambda x: x[0], reverse=True)
            self.index_dict = self.create_index_dict(sgo_results[0][1])

        self.cpt = sgo_results[0][2]
        self.build_cpt()
        self.SGO = sgo_results[0][1]
        if self.verbose:
            print("SGO Found")
            print(self.SGO)
            print(self.cpt)
        return None

    def build_cpt(self):
        l = []
        for i in range(len(self.cpt)):
            tmplist = defaultdict(float)
            for k in self.cpt[i].keys():
                tmplist[k] = self.cpt[i][k]
            l.append([tmplist[s] for s in self.states])
        self.cpt = np.array(l)

    def argmax(self, arr):
        return np.random.choice(np.argwhere(arr == np.amax(arr)).flatten(), size=1)[0]

    def predict(self, X_test):
        return [self.argmax((self.cpt[self.find_high(lag, self.index_dict)])) for i, lag in enumerate(X_test)]

    def return_probs(self, lag):
        state = self.find_high(lag, self.index_dict)
        if state < len(self.cpt):
            return self.cpt[self.find_high(lag, self.index_dict)]
        else:
            return [1/len(self.cpt) for _ in range(len(self.cpt))]

    def test(self, X_test, y_test):
        res = []
        for i, lag in enumerate(X_test):
            high = self.find_high(lag, self.index_dict)
            if high < len(self.cpt):
                if self.argmax((self.cpt[high])) == y_test[i]:
                    res.append(1)

        return sum(res) / len(y_test)

    def test_sample(self, x_test, y_test):
        pred = [choice(self.state_size, 1, p=self.cpt[self.find_high(lag, self.index_dict)])[0] for i, lag in
                enumerate(x_test)]
        return sum([1 for i in range(len(y_test)) if pred[i] == y_test[i]]) / len(y_test)
# %%
