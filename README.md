# CS 345 - Fall 2024 Course at the University of Nevada, Reno
#### Created by Samuel Mouradian - 4th Year CSE Major<br>

## <br>Final Project Details:
There are five files for this final project:
1. helpers.py
2. multiclass_classification.py
3. direct_multi-class_classification_with_resampling.py
4. performance_evaluation.py
5. evaluation_results.pdf


## <br>helpers.py
The "helpers.py" file is a collection of 3 functions that will load a file into a Panda Dataframe, which will then be cleaned, and finally separated into testing and training portion.<br>

- load_data(fname)
    - This function will take a filename from a predefined file directory path, and will load it into a Pandas DataFrame. The function will then return the DataFrame.
- clean_data(df)
    - This function will take the Pandas DataFrame, which we made using the previous function, and "clean" the data. We will remove non-numerical columns first, then replace all NaN/Inf values with median values. Finally, we will return the cleaned DataFrame.
- split_data(df)
    - This function will take the cleaned Pandas DataFrame and split it into 80% training data, as well as 20% testing. It will resample the DataFrame, then return 4 DataFrames: X_train, y_train, X_test, and y_test.<br>

## <br>multiclass_classification.py
The "multiclass_classification.py" file will have python code performing direct - and hierarchical - multi-class classification. There will also be a series of new functions that will be implemented:<br>

- direct_multiclass_train(model_name, X_train, y_train)
    - One will take a model and training data to form a trained model.
- direct_multiclass_test(model, X_test, y_test)
    - Another will evaluate the model using the test data, and will then produce an accuracy value.<br>


- improved_data_split(df)
    - The third implemented function will sample the data.
- get_binary_dataset(df)
    - The last two will, respectively, return the train and test dataframes, as well as convert "df" into a binary dataset and return it.<br>

## <br>direct_multi-class_classification_with_resampling.py
The "direct_multi-class_classification_with_resampling.py"<br>

## <br>performance_evaluation.py
The "performance_evaluation.py"<br>

## <br>evaluation_results.pdf
The "Performance_Evaluation.pdf" file will encapsulate the testing accuracies for tall the classification methods done in the "multiclass_classification.py" file. The results will be printed in this file with photo evidence that the code executed successfully.