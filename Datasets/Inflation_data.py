
import pandas as pd

class Inflation_data(object):

    @staticmethod
    def load_data():
        return pd.csv("data_files/inflation_data.csv")
