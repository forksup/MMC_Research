from Models.HMC import HMC


class FMC(HMC):
    def __init__(self, state_size, order):
        super().__init__(state_size, order=1)
        self.name = "FMC"
        self.states = self.possible_states.keys()

    def return_probs(self, lag):
        return self.transition_matrix[lag[-1]].toarray()[0]
