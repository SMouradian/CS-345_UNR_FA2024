# CS 345 - Fall 2024 Course at the University of Nevada, Reno
#### Created by Samuel Mouradian - 4th Year CSE Major<br>

## <br>Final Project Details:
There are four files for this final project:
1. helpers.py
2. multiclass_classification.py
4. performance_evaluation.py
5. evaluation_results.pdf<br>

# <br>NOTE - RUN "performance_evaluation.py" in order to generate results in "evaluations_results.pdf"!!<br>

## <br>helpers.py
The "helpers.py" file is a collection of 3 functions that will load a file into a Panda Dataframe, which will then be cleaned, and finally separated into testing and training portion.

- load_data(fname)
    - This function will take a filename from a predefined file directory path, and will load it into a Pandas DataFrame. The function will then return the DataFrame.
- clean_data(df)
    - This function will take the Pandas DataFrame, which we made using the previous function, and "clean" the data. We will remove non-numerical columns first, then replace all NaN/Inf values with median values. Finally, we will return the cleaned DataFrame.
- split_data(df)
    - This function will take the cleaned Pandas DataFrame and split it into 80% training data, as well as 20% testing. It will resample the DataFrame, then return 4 DataFrames: X_train, y_train, X_test, and y_test.<br>

## <br>multiclass_classification.py
The "multiclass_classification.py" file will have python code performing direct - and hierarchical - multi-class classification. The following function below will be implemented to do this:

- direct_multiclass_train(model_name, X_train, y_train)
    - This function will take a model and training data to form a trained model.
- direct_multiclass_test(model, X_test, y_test)
    - Another will evaluate the model using the test data, and will then produce an accuracy value.<br>

After the two functions above have been implemented, there will be a multi-class classification performed using MLP (or RFC). We will implement the following function to complete this portion:

- data_resampling(df, sampling_strategy)
    - This function will take the DataFrame and undersample it using the sampleing_strategy input. After, it will return the resampled df.<br>

Once the reasmpling is completed, we will use MLP (or RFC) to check the samples in order to find a malicious one. Once one has been found, we will perform multi-class classification again using MLP (or RFC) (without resampling) to identify the malicious activity present in the random forest. We will implement the following functions to help:

- improvedd_data_split(df)
    - This function will takes the original df into train and test sets that both contain all the categories. It will then return the two dataframes: df_train and df_test.
- get_binary_dataset(df)
    - This function will convert df into a binary dataset and return it.<br>

Once this is all completed, we can finally perform an evaluation with our functions.<br>

## <br>performance_evaluation.py
In this file representing our main code, we will evaluate the following:

- The testing_accuracy, classification_report, and confusion_matrix of <u>Direct Multi-Class Classification</u> using MLP and RFC, respectively.

- The The testing_accuracy, classification_report, and confusion_matrix of <u>Direct Multi-Class Classification</u> (__with resampling__) using MLP and RFC, respectively.

- The testing_accuracy, classification_report, and confusion_matrix of the <u>binary classification</u> (__with resampling__) in Hierarchical Multi-class Classification using MLP and RFC, respectively.

- The testing_accuracy, classification_report, and confusion_matrix of the <u>multi-class classification</u> in Hierarchical Multi-Class Classification using MLP and RFC, respectively.

These results will be posted, in single file, on a pdf document, and will be submitted along with the code files within a compressed zip folder.<br>

## <br>evaluation_results.pdf
This document will hold all the values created from the "performance_evaluation.py" file execution.