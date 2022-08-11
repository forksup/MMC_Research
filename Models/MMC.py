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

    def __init__(self, state_size, order, sgo_method: sgo_types = sgo_types.hillclimb, verbose = True):
        self.sgom = sgo_method
        self.state_size = state_size
        self.states = [i for i in range(state_size)]
        self.cpt = None
        self.SGO = None
        self.name = "MMC"
        self.verbose = verbose

    @staticmethod
    def find_high(lag, sgo=None):
        for s in sgo:
            if np.isin(s, lag):
                return s
        return None

    def geometric_mean(self):
        # use greedy for hillclimb
        # increase domain size too
        #
        sgo = []
        states_left_to_check = set(self.states)
        while len(states_left_to_check) > 1:
            states = []
            for s in states_left_to_check:
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
            s = self.find_high(lag, sgo)
            if s is not None:
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
        for s in sets:
            if l[0] in s or l[1] in s:
                s.add(l[0])
                s.add(l[1])
                return
        else:
            sets.append(set(l))

    def gen_prob(self, n, probs):
        prod = 1
        for kn, d in n.items():
            for key, value in d.items():
                prod += n[kn][key] * math.log(probs[kn][key])
        return prod

    def gen_prob_dict(self, n, probs):
        result = defaultdict(dict)
        prod = 1
        for kn, d in n.items():
            for key, value in d.items():
                result[kn][key] = n[kn][key] * math.log(probs[kn][key]+1)
        return result

    def find_SGO_greedy(self):
        final_SGO = ""
        states_to_check = set(self.states) - set([-1])
        SGO = []

        results = []
        for s in states_to_check:
            n, prob = self.calculate_probabilities([s])
            tony = self.gen_prob_dict(n,prob)[s]
            results.append((max(tony.values()), s))

        while len(states_to_check) > 1:
            results = []
            for s in states_to_check:
                n, prob = self.calculate_probabilities(SGO+[s])
                tony = self.gen_prob_dict(n,prob)[s]
                results.append((max(tony.values()), s))
            results.sort(reverse=True)
            s = results[0][1]
            states_to_check.remove(s)
            SGO.append(s)
        SGO.append(states_to_check.pop())
        return SGO

    def train(self, X_train, y_train):

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
        if len(SGOs) > 1:
            for SGO in SGOs:
                count_dict, n = self.calculate_probabilities(SGO)
                sgo_results.append((self.gen_prob(count_dict, n), SGO, deepcopy(n)))
                """
                genprobs = self.gen_prob_dict(count_dict, n)
                #self.cpt = n
                #self.build_cpt()
                #print(self.cpt)
    
                max_probs = defaultdict(float)
                for k, val in genprobs.items():
                    max_probs[k] = max(val.values())
    
                mis_alignment = []
    
                # Retrieve all combinations of any two states to check for misalignments
                for l in list(itertools.combinations([s for s in self.states if s != -1], 2)):
                    a = SGO.index(l[1]) - SGO.index(l[0])
                    b = max_probs[l[0]] - max_probs[l[1]]
                    if np.sign(a) != np.sign(b):
                        self.combine_misalignments(l, mis_alignment)
                if self.verbose:
                    print(SGO)
                    print(mis_alignment)
                for m in mis_alignment:
                    # Gather all probability values for all misalligned states
                    input_table = {s: count_dict[s] for s in m}
                    new_table = deepcopy(input_table)
    
                    # Combine all the cpt's
                    res = dict(sum((Counter(dict(x)) for x in input_table.values()), Counter()))
    
                    max_key = max(res, key=res.get)
                    max_prob = res[max_key] / sum(res.values())
                    rem = 1 - max_prob
                    for key in new_table:
                        new_table[key][max_key] = max_prob
                        running_sum = max_prob
                        tot = sum(new_table[key].values()) - running_sum
    
                        for k2 in dict(sorted(new_table[key].items(), key=lambda item: item[1], reverse=True)):
    
                            if k2 == max_key:
                                continue
    
                            if new_table[key][k2] == 0:
                                lval = [k3 for k3, v2 in count_dict[key].items() if v2 == 0]
                                probtoset = rem / len(lval)
                                for ktset in lval:
                                    new_table[key][ktset] = probtoset
                                break
    
                            nns = (new_table[key][k2]/tot)*rem
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
                    for key in new_table:
                        n[key] = new_table[key]
                if self.verbose:
                    for key in n:
                        print(sum(n[key].values()))
                """
        sgo_results.sort(key=lambda x: x[0], reverse=True)
        self.cpt = sgo_results[0][2]
        self.SGO = sgo_results[0][1]
        self.build_cpt()
        if self.verbose:
            print("SGO Found")
            print(self.SGO)
            print(self.cpt)
        return self.test(X_train, y_train)

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



    def test(self, X_test, y_test):

        return sum([1 for i, lag in enumerate(X_test)
                    if self.argmax((self.cpt[self.find_high(lag, self.SGO)]))
                        == y_test[i]]) \
               / len(y_test)

    def test_sample(self, X_test, y_test):
        pred = [choice(self.state_size, 1, p=self.cpt[self.find_high(lag, self.SGO)])[0] for i, lag in
                enumerate(X_test)]
        return sum([1 for i in range(len(y_test)) if pred[i] == y_test[i]]) / len(y_test)
# %%
