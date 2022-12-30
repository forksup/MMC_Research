import pandas as pd


class Inflation_Data(object):

    @staticmethod
    def load_data():
        return pd.csv("data_files/inflation_data.csv")
