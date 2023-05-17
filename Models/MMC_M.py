from enum import Enum
from Models.MMC import MMC, sgo_types

# An MMC model with combining misalignment feature disabled
class MMC_M(MMC):
    def __init__(
        self,
        state_size,
        order,
        sgo_method: sgo_types = sgo_types.greedy,
        verbose=False,
    ):
    
        super(state_size, order, sgo_method, False).__init__()

