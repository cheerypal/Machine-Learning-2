from sklearn import metrics
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt


# This function gets the ROC area for each of the classes using the binary classes.
# Inputs : prediction - this is the predicted classes from the classifier
# Inputs : FileType - this is the type of test that is ran. The types of test available are
#          'train' for testing on the training data provided
#          'test' for testing on the testing data provided
#          '4000' for testing using the test files that have 4000 extra instances
#          '9000' for testing using the test files that have 9000 extra instances
# Outputs : dataframe containing all the ROC areas for each class using the class specific file.
def get_ROC_AREA(prediction, fileType):
    auc_arr = list()
    # loop through all 10 files
    for i in range(0, 10):
        # convert prediction into a boolean list
        bin_pred = (prediction == i)
        # invert the list so that it matches the provided files
        bin_pred = [not value for value in bin_pred]
        # only access when 4000/9000 instance data is being used.
        if fileType != "4000" and fileType != "9000":
            auc_arr.append(metrics.roc_auc_score(pd.read_csv("../" + fileType + "ing_data/y_" + fileType + "_smpl_" +
                                                             str(i) + ".csv"), bin_pred))
        else:

            # generate Area under curve using the file and binary predictions.
            auc_arr.append(metrics.roc_auc_score(pd.read_csv("../" + fileType + "_data/y_test_smpl_" +
                                                             str(i) + ".csv_" + fileType + ".csv"), bin_pred))
    # return dataframe with roc results.
    return pd.DataFrame(data=auc_arr, columns=["ROC Area"])


# This function gets the ROC tpr, fpr and outputs ROC plots using the binary class files.
# Inputs : prediction - this is the predicted classes from the classifier
# Inputs : FileType - this is the type of test that is ran. The types of test available are
#          'train' for testing on the training data provided
#          'test' for testing on the testing data provided
#          '4000' for testing using the test files that have 4000 extra instances
#          '9000' for testing using the test files that have 9000 extra instances
# Inputs : Visualise - this is a boolean param when True it will plot the ROC curve
# Outputs : Dataframe containing the Average TPR and FPR for each class
# Outputs : ROC curve plot if Visualise is True
def get_TPR_FPR(prediction, fileType, visualise):
    rates = list()
    # Loop for all 10 files.
    for i in range(0, 10):
        # convert prediction into a boolean list
        bin_pred = (prediction == i)
        # invert the list so that it matches the provided files
        bin_pred = [not value for value in bin_pred]
        # only access when 4000/9000 instance data is being used.
        if fileType != "4000" and fileType != "9000":
            fpr, tpr, thresholds = metrics.roc_curve(
                pd.read_csv("../" + fileType + "ing_data/y_" + fileType + "_smpl_" +
                            str(i) + ".csv"), bin_pred)

            # append mean rates to rate array
            rates.append([np.mean(tpr), np.mean(fpr)])

        else:

            # generate tpr, fpr using the test file and the binary prediction list
            fpr, tpr, thresholds = metrics.roc_curve(pd.read_csv("../" + fileType + "_data/y_test_smpl_" +
                                                                 str(i) + ".csv_" + fileType + ".csv"), bin_pred)

            rates.append([np.mean(tpr), np.mean(fpr)])

        # plot chart if true  of the ROC curve
        if visualise:
            plot_roc_curve(i, fpr, tpr)

    # Return Dataframe contain average tpr and fpr
    return pd.DataFrame(data=rates, columns=["Average tpr", "Average fpr"])


# plots ROC curve given FPR and TPR values
def plot_roc_curve(label, fpr, tpr):
    title = "Rate for class " + str(label)
    plt.figure(figsize=(8, 6))
    plt.plot(fpr, tpr)
    plt.axis([0, 1, 0, 1])
    plt.xlabel("False Positive Rate")
    plt.ylabel("True Positive Rate")
    plt.suptitle(title)
    plt.grid(True)
    plt.show()


# Question 3
# Classifier tester uses training data and testing data to generate:
# Precision, Recall, F1-Measure, ROC Curves, ROC Area and Accuracy
# Inputs : classifier - this is the classifier to be using on the training and testing data :
#           Tree, Linear Regression MLP
# Inputs : testType - this is the test type used for calculating ROC metrics
#          'train' for testing on the training data provided
#          'test' for testing on the testing data provided
#          '4000' for testing using the test files that have 4000 extra instances
#          '9000' for testing using the test files that have 9000 extra instances
# Inputs : train_data, train_labels, testData, testLabels : these are the data files used in testing
# Inputs : Visualise - this is a boolean param when True it will plot the ROC curve
# Outputs : Precision, Recall, F1-Measure, ROC Curves, ROC Area and Accuracy
def classifier_tester(classifier, testType, train_data, train_labels, testData, testLabels, visualise):
    print("Starting .... \n")
    classifier.fit(train_data, train_labels)

    # predicts the labels from the test data given
    pred = classifier.predict(testData)
    print("Starting Confusion Matrix ....")
    # plots confusion matrix between testingLabels and the predicted labels
    print("\nConfusion Matrix\n", metrics.confusion_matrix(testLabels, pred))
    print("\n")
    print("\n", metrics.classification_report(testLabels, pred))

    # get tpr, fpr and ROC area
    print("\n", get_TPR_FPR(pred, fileType=testType, visualise=visualise))
    print("\n", get_ROC_AREA(pred, fileType=testType))

    accuracy = metrics.accuracy_score(testLabels, pred)
    print("\nAccuracy: " + str(accuracy))

