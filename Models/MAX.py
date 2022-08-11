from random import randint
import numpy as np
from datetime import datetime
from Models.model_sources.markov_source import MarkovChain


class MAX(object):

    def __init__(self, state_size, order=3):
        self.state_size = state_size
        self.order = order
        self.name = "MAX"
        self.models = [MarkovChain(state_size, 1) for _ in range(order)]

    def train(self, X_train, y_train):
        for i, m in enumerate(self.models):
            m.fit([X_train[:, -(i + 1)], y_train])
        return self.test(X_train, y_train)

    def test(self, X_test, y_test):
        # when preforming this technique accuracy decreases due to shrinking testing dataset
        acc = []
        slices = []
        count = 0
        start = datetime.now()

        for x, y in zip(X_test, y_test):
            if -1 == y:
                continue
            if -1 in x:
                results = [[0, randint(0, self.state_size)]]
                count += 1
            else:
                results = []

                for key, m in enumerate(self.models[:self.order + 1]):
                    # print(episode[i - 1 - key:i - key])
                    probs = m.transition_matrix[m.possible_states[tuple([x[-(key + 1)]])]]
                    arg = np.argmax(probs)

                    # probability, state,  slice
                    results.append((probs[0, arg], arg, key))
                results.sort(reverse=True)
                slices.append(results[0][2])

            lst = [r[1] for r in results]
            exp = max(set(lst), key=lst.count)

            if exp == y:
                acc.append(1)
            else:
                acc.append(0)

        return sum(acc) / len(acc)
