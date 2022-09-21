from Models.HMC import HMC


class FMC(HMC):

    def __init__(self, state_size, order):
        super().__init__(state_size, order=1)
        self.name = "FMC"
