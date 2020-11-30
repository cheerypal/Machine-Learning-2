from sklearn import metrics
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt


# get ROC using the binary files
# fileType can either be "train" or "test"
# returns data frame containing the ROC area for each binary file
def get_ROC_AREA(prediction, fileType):
    auc_arr = list()
    for i in range(0, 10):
        if fileType != "4000" and fileType != "9000":
            auc_arr.append(metrics.roc_auc_score(pd.read_csv("../" + fileType + "ing_data/y_" + fileType + "_smpl_" +
                                                             str(i) + ".csv"), prediction))
        else:
            auc_arr.append(metrics.roc_auc_score(pd.read_csv("../" + fileType + "_data/y_test_smpl_" +
                                                             str(i) + ".csv_" + fileType + ".csv"), prediction))
    return pd.DataFrame(data=auc_arr, columns=["ROC Area"])


# get tpr and fpr rate for certain binary files.
# fileType can either be "train" or "test"
# returns data frame containing the True positive rate and false positive rate for each binary file
def get_TPR_FPR(prediction, fileType, visualise):
    rates = list()

    for i in range(0, 10):
        if fileType != "4000" and fileType != "9000":
            fpr, tpr, thresholds = metrics.roc_curve(
                pd.read_csv("../" + fileType + "ing_data/y_" + fileType + "_smpl_" +
                            str(i) + ".csv"), prediction)

            rates.append([np.mean(tpr), np.mean(fpr)])
            if visualise:
                plot_roc_curve(i, fpr, tpr)

        else:
            fpr, tpr, thresholds = metrics.roc_curve(pd.read_csv("../" + fileType + "_data/y_test_smpl_" +
                                                                 str(i) + ".csv_" + fileType + ".csv"), prediction)

            rates.append([np.mean(tpr), np.mean(fpr)])
            if visualise:
                plot_roc_curve(i, fpr, tpr)

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
# test method for when test data needs to be used.
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

