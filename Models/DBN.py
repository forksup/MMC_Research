from Models.HMC import HMC


class DBN(HMC):

    def __init__(self, state_size, order):
        super().__init__(state_size, order=1)
        self.name = "DBN"
