# Step 1 - Import the necessary packages.
import pandas as pd
import numpy as np


# Step 2 - Create the *load_data(fname)* function.
#   This function will take a filename from a predefined file directory path,
#   and will load it into a Pandas DataFrame. The function will then return
#   the DataFrame.
def load_data(fname):
    folder_path = open('./CS-345_UNR_FA2024/')  # opens csv file directory
    df = pd.read_csv(folder_path + fname)  # reads the specified file and loads it into the Pandas DataFrame
    return df  # returns the Pandas DataFrame


# Step 3 - Create the *clean_data(df)* function.
#   This function will take the Pandas DataFrame, which we made using the
#   previous function, and "clean" the data. We will remove non-numerical
#   columns first, then replace all NaN/Inf values with median values.
#   Finally, we will return the cleaned DataFrame.
def clean_data(df):
    df = df.select_dtypes(include = [np.number])  # removes non-numeric columns from the data
    df.replace([np.inf, -np.inf], np.nan, inplace = True)  # replaces Inf values with NaN values

    df.fillna(df.median(), inplace = True)  # replaces NaN values with median values

    return df  # returns the cleaned DataFrame


# Step 4 - Create the *split_data(df)* function.
#   This function will take the cleaned Pandas DataFrame and split it into
#   80% training data, as well as 20% testing. It will resample the DataFrame,
#   then return 4 DataFrames: X_train, y_train, X_test, and y_test.
def split_data(df):
    # split the data into 80% for training and 20% for testing
    mask = np.random.rand(len(df)) < 0.8
    df_train = df[mask]
    df_test = df[~mask]
    
    # 80% set to training data
    X_train = df_train[df_train.columns[:-1]]
    y_train = df_train[df_train.columns[-1]]

    # 20% set to testing data
    X_test = df_test[df_test.columns[:-1]]
    y_test = df_test[df_test.columns[-1]]

    # return all 4 separated DataFrames
    return X_train, y_train, X_test, y_test