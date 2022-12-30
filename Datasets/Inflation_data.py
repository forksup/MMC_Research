import pandas as pd


class inflationData(object):

    @staticmethod
    def load_data():
        return pd.csv("data_files/inflation_data.csv")
