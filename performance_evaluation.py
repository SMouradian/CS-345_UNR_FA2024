# Step 1 - Import the necessary packages/files.
# Files
from helpers import pd, np, load_data, clean_data, split_data
from multiclass_classification import direct_multiclass_test, direct_multiclass_train, data_resampling, improved_data_split, get_binary_dataset
from multiclass_classification import RandomForestClassifier, MLPClassifier, accuracy_score, classification_report
# Packages
from sklearn.metrics import ConfusionMatrixDisplay, confusion_matrix
from sklearn.preprocessing import LabelEncoder
import matplotlib.pyplot as plt



# COMPLETE THE PERFORMANCE EVALUATION BELOW. WRITE A MAIN CODE FILE THAT
#   UTILIZES ALL THE FUNCTIONS TO PERFORM THE EVALUATIONS DEFINED IN THE
#   README DOCUMENT.

with open('evaluations_reults.pdf', 'w') as f:
    def main():
        # Step 1 - Evaluate the testing_accuracy, classification_report, and confusion_matrix of
        #   Direct Multi-Class Classification using MLP and RFC, respectively.
        df = load_data('MachineLearningCVE')
        df_clean = clean_data(df)
    
        X_train, y_train, X_test, y_test = split_data(df_clean)
    
        # Direct Multi-Class Classification without resampling
        model = direct_multiclass_train("rfc", X_train, y_train)
        direct_multiclass_test(model, X_test, y_test)

        # accuracy and classification report for evaluation 1
        pred = model.predict(X_test)
        acc = accuracy_score(pred, y_test)
        print('Testing accuracy: {:.5f}'.format(acc) )
        print('Classification_report:')
        print(classification_report(y_test, pred))

        # confusion_matrix of evaluation 1
        label_encoder = LabelEncoder()
        label_encoder.fit(y_test)  # Fit the encoder based on the true labels
        labels = label_encoder.classes_  # Now, map the numerical indices to the original label names
        cm = confusion_matrix(y_test, pred)  # Generate the confusion matrix
        cd = ConfusionMatrixDisplay(confusion_matrix=cm, display_labels=labels)  # Plot the confusion matrix with the correct labels
        cd.plot()
        plt.xticks(rotation=90)
        plt.show()

    
        # Direct Multi-Class Classification with resampling
        df_resampled = data_resampling(df_clean, "undersample")
        X_train_resampled, y_train_resampled, X_test_resampled, y_test_resampled = split_data(df_resampled)
        model_resampled = direct_multiclass_train("mlp", X_train_resampled, y_train_resampled)
        direct_multiclass_test(model_resampled, X_test_resampled, y_test_resampled)

        # accuracy and classification report for evaluation 2
        pred = model_resampled.predict(X_test)
        acc = accuracy_score(pred, y_test)
        print('Testing accuracy: {:.5f}'.format(acc) )
        print('Classification_report:')
        print(classification_report(y_test, pred))

        # confusion_matrix of evaluation 2
        label_encoder = LabelEncoder()
        label_encoder.fit(y_test)  # Fit the encoder based on the true labels
        labels = label_encoder.classes_  # Now, map the numerical indices to the original label names
        cm = confusion_matrix(y_test, pred)  # Generate the confusion matrix
        cd = ConfusionMatrixDisplay(confusion_matrix=cm, display_labels=labels)  # Plot the confusion matrix with the correct labels
        cd.plot()
        plt.xticks(rotation=90)
        plt.show()


        # Hierarchical Multi-Class Classification
        df_binary = get_binary_dataset(df_clean)
        df_train, df_test = improved_data_split(df_binary)
        X_train_bin, y_train_bin, X_test_bin, y_test_bin = split_data(df_train)
        model_bin = direct_multiclass_train("rfc", X_train_bin, y_train_bin)
        direct_multiclass_test(model_bin, X_test_bin, y_test_bin)

        # accuracy and classification report for evaluation 3
        pred = model_bin.predict(X_test)
        acc = accuracy_score(pred, y_test)
        print('Testing accuracy: {:.5f}'.format(acc) )
        print('Classification_report:')
        print(classification_report(y_test, pred))
        
        # confusion_matrix of evaluation 3
        label_encoder = LabelEncoder()
        label_encoder.fit(y_test)  # Fit the encoder based on the true labels
        labels = label_encoder.classes_  # Now, map the numerical indices to the original label names
        cm = confusion_matrix(y_test, pred)  # Generate the confusion matrix
        cd = ConfusionMatrixDisplay(confusion_matrix=cm, display_labels=labels)  # Plot the confusion matrix with the correct labels
        cd.plot()
        plt.xticks(rotation=90)
        plt.show()


        df_malicious = df_test[df_test['target'] == 1]  # Test malicious samples
        X_train_mal, y_train_mal, X_test_mal, y_test_mal = split_data(df_malicious)
        model_mal = direct_multiclass_train("mlp", X_train_mal, y_train_mal)
        direct_multiclass_test(model_mal, X_test_mal, y_test_mal)

        # accuracy and classification report for evaluation 4
        pred = model_mal.predict(X_test)
        acc = accuracy_score(pred, y_test)
        print('Testing accuracy: {:.5f}'.format(acc) )
        print('Classification_report:')
        print(classification_report(y_test, pred))

        # confusion_matrix of evaluation 4
        label_encoder = LabelEncoder()
        label_encoder.fit(y_test)  # Fit the encoder based on the true labels
        labels = label_encoder.classes_  # Now, map the numerical indices to the original label names
        cm = confusion_matrix(y_test, pred)  # Generate the confusion matrix
        cd = ConfusionMatrixDisplay(confusion_matrix=cm, display_labels=labels)  # Plot the confusion matrix with the correct labels
        cd.plot()
        plt.xticks(rotation=90)
        plt.show()