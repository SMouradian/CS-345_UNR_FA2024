'''import pandas as pd
import numpy as np

def load_data(fname):
    folder_name = open('./CS-345_UNR_FA2024/MachineLearningCVE/')  # opens csv file directory
    df = pd.read_csv(folder_name + '{fname}')  # reads the specified file and loads it into the Pandas DataFrame
    return df  # returns the Pandas DataFrame

def clean_data(df):
    df = df.select_dtypes(include = [np.number])  # removes non-numeric columns from the data
    df.replace([np.inf, -np.inf], np.nan, inplace = True)  # replaces Inf values with NaN values

    df.fillna(df.median(), inplace = True)  # replaces NaN values with median values

    return df  # returns the cleaned DataFrame

def split_data(df):
    resampled_df = pd.DataFrame(columns=df.columns)  # resampled the Pandas DataFrame

    # split the data into 80% for training and 20% for testing
    mask = np.random.rand(len(resampled_df)) < 0.8
    df_train = resampled_df[mask]
    df_test = resampled_df[~mask]
    
    # 80% set to training data
    X_train = df_train[df_train.columns[:-1]]
    y_train = df_train[df_train.columns[-1]]

    # 20% set to testing data
    X_test = df_test[df_test.columns[:-1]]
    y_test = df_test[df_test.columns[-1]]

    # return all 4 separated DataFrames
    return X_train, y_train, X_test, y_test '''