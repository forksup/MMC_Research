import numpy as np
from Models.model_sources.path_encoder import SequenceCutter, PathEncoder


class financial_data(object):
    @staticmethod
    def gen_data(states, order, size, verbose=False):
        import requests
        import pandas as pd

        interval = "1h"
        symbol = "ETH-USD"

        def get_crypto_price(symbol, exchange, interval, start_date=None):
            api_url = f"https://eodhistoricaldata.com/api/intraday/{symbol}.CC?api_token=&interval={interval}&fmt=json"
            raw_df = requests.get(api_url).json()
            df = pd.DataFrame(raw_df)
            df["close"] = df["close"].astype(float)
            return df

        df = get_crypto_price(symbol=symbol, exchange="USD", interval=interval)
        df["Change"] = df["close"].diff()

        df["Change_enc"] = np.nan
        std = df.std()["Change"]

        df.loc[df.Change < 0.0, "Change_enc"] = "drop"
        df.loc[df.Change <= -std, "Change_enc"] = "big_drop"
        df.loc[df.Change <= -2 * std, "Change_enc"] = "bigger_drop"
        df.loc[df.Change <= -3 * std, "Change_enc"] = "biggest_drop"

        df.loc[df.Change > 0.0, "Change_enc"] = "rise"
        df.loc[df.Change >= std, "Change_enc"] = "big_rise"
        df.loc[df.Change >= 2 * std, "Change_enc"] = "bigger_rise"
        df.loc[df.Change >= 3 * std, "Change_enc"] = "biggest_rise"

        # New York
        # London
        # Asia

        # df.loc[df.Change.between(-5, 5), 'Change_enc'] = 'NO_CHANGE'

        df.dropna(inplace=True)
        sc = SequenceCutter(order)
        x, y = sc.transform(df.Change_enc.values)

        pe = PathEncoder(order)
        pe.fit(x, y)

        x_tr, y_tr = pe.transform(x, y)
        div = round(len(x_tr) * 0.85)
        # (X_train, X_test, y_train, y_test) = train_test_split(x_tr, y_tr)
        return x_tr[div:, :], x_tr[:div, :], y_tr[div:], y_tr[:div]
