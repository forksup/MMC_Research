from random import randint
from Models.model_sources.markov_source import MarkovChain
import numpy as np
from datetime import datetime


class HMC(MarkovChain):

    def __init__(self, state_size, order=3, verbose=True):
        self.state_count = state_size
        self.order = order
        self.verbose = verbose
        self.name = "HMC"
        super().__init__(self.state_count, order=order, verbose=verbose)

    def train(self, X_train, y_train):
        super().fit([X_train[:, -self.order:], y_train])
        return None

    def test(self, X_test, y_test):
        results = [1 for i in range(len(X_test)) if np.argmax(self.transition_matrix[self.possible_states[tuple(X_test[i][-self.order:])]]) == y_test[i]]
        return sum(results) / len(X_test)
