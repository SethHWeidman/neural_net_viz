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
    X1, Y1 = make_classification(n_features=input_neurons,
                                 n_redundant=0,
                                 n_informative=1,
                                 n_clusters_per_class=1)

    x_cols = ['X' + str(x) for x in range(input_neurons)]
    all_cols = x_cols + ["Y"]

    df = pd.DataFrame(np.concatenate((X1, Y1.reshape(100, 1)), axis=1),
                      columns=all_cols)

    mms = MinMaxScaler(feature_range=(1, 2))
    df[all_cols] = mms.fit_transform(df[all_cols])

    add_data_to_db(df.to_json(), x_cols)

    return df

def add_data_to_db(df, x_cols):
    '''
    Add necessary data to database
    '''
    db.add_key("dataframe", df)
    db.add_key("x_cols", x_cols)

    return True
