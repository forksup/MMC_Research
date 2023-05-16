from enum import Enum
from Models.MMC import MMC


class sgo_types(Enum):
    greedy = 1
    hillclimb = 2
    full = 3
    geometric_mean = 4


class MMC(MMC):
    def __init__(
        self,
        state_size,
        order,
        sgo_method: sgo_types = sgo_types.greedy,
        misalignment=True,
        verbose=False,
    ):
        self.sgom = sgo_method
        self.state_size = state_size
        self.states = [i for i in range(state_size)]
        self.cpt = None
        self.SGO = None
        self.name = "MMC"
        self.verbose = verbose
        self.index_dict = {}
