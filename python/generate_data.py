from sklearn.datasets import make_classification
from sklearn.preprocessing import MinMaxScaler
import pandas as pd
import numpy as np
import os.path
import sys
sys.path.append(os.path.join(os.path.dirname(__file__), '.'))
import database as db

def generate_data(input_neurons):
    '''
    Generate data based on the number of input neurons
    '''
    num_obs = 100
    X1, Y1 = make_classification(n_samples=num_obs,
                                 n_features=input_neurons,
                                 n_redundant=0,
                                 n_informative=1,
                                 n_clusters_per_class=1)

    x_cols = ['X' + str(x) for x in range(input_neurons)]
    all_cols = x_cols + ["Y"]

    df = pd.DataFrame(np.concatenate((X1, Y1.reshape(100, 1)), axis=1),
                      columns=all_cols)

    mms = MinMaxScaler(feature_range=(1, 2))
    df[all_cols] = mms.fit_transform(df[all_cols])

    db.create_key("dataframe", df)
    db.create_key("x_cols", x_cols)
    db.create_key("num_obs", num_obs)

    return df


def read_x_row():

    df_x = db.read_key('dataframe')[db.read_key('x_cols')]
    samp = np.random.randint(0, db.read_key('num_obs'))
    x_row = np.array(df_x.iloc[samp, :], ndmin=2)

    return x_row
