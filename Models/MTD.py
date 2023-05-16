from Models.model_sources.markov_source import MarkovChain
from datetime import datetime
from random import randint
import numpy as np


class CAP(MarkovChain):
    name = "CAP"

    def __init__(self, state_size, order):
        self.state_size = state_size
        super().__init__(self.state_size)
        self.order = order

        self.cpt = None
        self.SGO = None

    def train(self, X_train, y_train):
        for i in range(self.order):
            super().fit([X_train[:, -(i + 1)], y_train], False)
        super().normalize_transitions()
        return None

    def test(self, X_test, y_test):
        # when preforming this technique accuracy decreases due to shrinking testing dataset
        acc = []
        slices = []
        start = datetime.now()

        for x, y in zip(X_test, y_test):
            if -1 == y:
                continue
            if -1 in x:
                results = [[0, randint(0, self.state_size)]]
            else:
                results = []

                for key in range(self.order):
                    probs = self.transition_matrix[
                        self.possible_states[tuple([x[-(key + 1)]])]
                    ]
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
