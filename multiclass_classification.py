# Step 1 - Import the necessary packages/files.
# Files
from helpers import pd, np, load_data, clean_data, split_data
# Packages
from sklearn.ensemble import RandomForestClassifier
from sklearn.neural_network import MLPClassifier
from imblearn.under_sampling import RandomUnderSampler
from sklearn.metrics import accuracy_score, classification_report


# Step 2 - Create the direct_multiclass_train(model_name, X_train, y_train)
# function.
#   This function will take a model and training data to form a trained model.
def direct_multiclass_train(model_name, X_train, y_train):
    if model_name == 'rfc':
        rfc = RandomForestClassifier().fit(X_train, y_train)  # if 'rfc' is entered, make a trained RFC of the data.
    elif model_name == 'mlp':
        mlp = MLPClassifier(hidden_layer_sizes=(40,), random_state=1,
                            max_iter=300).fit(X_train, y_train)  # if 'mlp' is entered, the code should make a trained MLP of the data.


# Step 3 - Create the direct_multiclass_test(model, X_test, y_test) function.
#   Another will evaluate the model using the test data, and will then produce
#   an accuracy value.
def direct_multiclass_test(model, X_test, y_test):
    pred = model.predict(X_test)
    acc = accuracy_score(pred, y_test)  # creates an accuracy score off the prediction of the rfc/mlp model

    return acc  # returns the accuracy score


# Step 4 - Complete Direct Multi-Class Classficiation (with Resampling).
#   Perform data resampling to handle the unbalanced data distribution, then
#   conduct multi-class classification using MLP (or RFC). Implement the
#   data_resampling(df, sampling_strategy) function to complete this objective.
def data_resampling(df, sampling_strategy):
    rus = RandomUnderSampler(sampling_strategy = {'BENIGN': 10000, 'DDoS': 10000, 'DoS Hulk': 10000, 'PortScan': 10000
                                              }, random_state = 2)  # create a RandomUnderSampler
    
    X = df[df.columns[:-1]]
    y = df[df.columns[-1]]
    
    X_resampled, y_resampled = rus.fit_resample(X,y)  # create resampled X and y values for the DataFrame
    
    resampled_df = pd.DataFrame(columns=df.columns)  # create the resampled DataFrame

    # include the resampled X and y valued into the DataFrame columns
    resampled_df[resampled_df.columns[:-1]] = X_resampled
    resampled_df[resampled_df.columns[-1]] = y_resampled

    return resampled_df


# Step 5 - Complete Hierarchical Multi-Class Classification
#   Perform a check for malicous samples, then perform multi-class classification
#   (without resampling) using MLP (or RFC) to identify the malicious activity occurring.
#   Implement the improved_data_split(df) and get_binary_dataset(df) functions
#   to do this.
def improved_data_split(df):
    label_set = set(df['Label'])  # begins creating the categories

    train_df_list = []
    test_df_list = []
    
    for label in label_set:
        mask = np.random.rand(len(df[df['Label'] == label])) < 0.8  # splits the original data into test and train sets
        print('num of samples for "{}": {}'.format(label, len(mask)))
        train_df_list.append(
            df[df['Label'] == label][mask]  # gives df_train all categories
        )
        test_df_list.append(
            df[df['Label'] == label][~mask] # gives df_test all categories
        )
        
    df_train = pd.concat(train_df_list)  # updates structure of df_train
    df_test = pd.concat(test_df_list)  # updates structure of df_test

    return df_test, df_train  # returns df_train and df_test


def get_binary_dataset(df):
    df_binary = df.copy()  # creates binary verion of original data

    # convert attack labels to "MALICIOUS":
    df_binary.loc[df_binary['Label'] != 'BENIGN', 'Label'] = 'MALICIOUS'

    return df_binary