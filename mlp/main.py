from sklearn.neural_network import MLPClassifier
from sklearn.datasets import make_classification
from sklearn.model_selection import train_test_split, cross_val_predict
from sklearn.metrics import confusion_matrix, accuracy_score, precision_score, recall_score, f1_score, roc_curve
from sklearn import metrics
import numpy
import matplotlib.pyplot as plt

import pandas
data = pandas.read_csv("../training_data/x_train_gr_smpl.csv")
labels = pandas.read_csv("../training_data/y_train_smpl.csv")
testingData = pandas.read_csv("../testing_data/x_test_gr_smpl.csv")
testingLabels = pandas.read_csv("../testing_data/y_test_smpl.csv")

print("MLP classifier starting")
clf = MLPClassifier(hidden_layer_sizes=(15,), random_state=1, max_iter=100, warm_start=False)
print("MLP classifier finished")

def start_mlp(clf):

    print("Cross val starting")
    label_predictions = cross_val_predict(clf, data, labels["0"], cv=10)
    print("Cross val finished")

    print("Building confusion matrix")
    confusion_matrix = metrics.confusion_matrix(labels, label_predictions)
    print("confusion matrix complete")


    print(confusion_matrix)

    # get precision, recall and f1 measure
    print("\n", metrics.classification_report(labels, label_predictions))

    # get tpr, fpr and ROC area
    print("\n", get_TPR_FPR(label_predictions, "train", visualise=True))
    print("\n", get_ROC_AREA(label_predictions, "train"))

    # Get accuracy of the cross validation
    accuracy = accuracy_score(labels, label_predictions)
    print("\nAccuracy: " + str(accuracy))




# get ROC using the binary files
# fileType can either be "train" or "test"
# returns data frame containing the ROC area for each binary file
def get_ROC_AREA(prediction, fileType):
    auc_arr = list()
    for i in range(0, 10):
        if fileType != "4000" and fileType != "9000":
            auc_arr.append(metrics.roc_auc_score(pandas.read_csv("../" + fileType + "ing_data/y_" + fileType + "_smpl_" +
                                                             str(i) + ".csv"), prediction))
        else:
            auc_arr.append(metrics.roc_auc_score(pandas.read_csv("../" + fileType + "_data/y_test_smpl_" +
                                                             str(i) + ".csv_" + fileType + ".csv"), prediction))
    return pandas.DataFrame(data=auc_arr, columns=["ROC Area"])


# get tpr and fpr rate for certain binary files.
# fileType can either be "train" or "test"
# returns data frame containing the True positive rate and false positive rate for each binary file
def get_TPR_FPR(prediction, fileType, visualise):
    rates = list()

    for i in range(0, 10):
        if fileType != "4000" and fileType != "9000":
            fpr, tpr, thresholds = metrics.roc_curve(
                pandas.read_csv("../" + fileType + "ing_data/y_" + fileType + "_smpl_" +
                            str(i) + ".csv"), prediction)

            rates.append([numpy.mean(tpr), numpy.mean(fpr)])
            if visualise:
                plot_roc_curve(i, fpr, tpr)

        else:
            fpr, tpr, thresholds = metrics.roc_curve(pandas.read_csv("../" + fileType + "_data/y_test_smpl_" +
                                                                 str(i) + ".csv_" + fileType + ".csv"), prediction)

            rates.append([numpy.mean(tpr), numpy.mean(fpr)])
            if visualise:
                plot_roc_curve(i, fpr, tpr)

    return pandas.DataFrame(data=rates, columns=["Average tpr", "Average fpr"])




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
def mlp_test_data(clf, testType, testData, testLabels, visualise):
    print("Starting .... \n")

    clf = clf.fit(data, labels["0"])

    # predicts the labels from the test data given
    pred = clf.predict(testData)
    print("Starting Confusion Matrix ....")
    # plots confusion matrix between testingLabels and the predicted labels
    print("\nConfusion Matrix\n", metrics.confusion_matrix(testLabels, pred))
    print("\n")
    print("\n", metrics.classification_report(testLabels, pred))

    # get tpr, fpr and ROC area
    print("\n", get_TPR_FPR(pred, fileType=testType, visualise=visualise))
    print("\n", get_ROC_AREA(pred, fileType=testType))

    accuracy = accuracy_score(testLabels, pred)
    print("\nAccuracy: " + str(accuracy))





start_mlp(clf)


# Question 3
print("\nTesting using dataset testing data ....\n")
mlp_test_data(clf, "test", testingData, testingLabels, visualise=True)

# Question 4
print("\nTesting using 4000 moved testing data ....\n")
test_4000 = pandas.read_csv("../4000_data/x_test_gr_smpl.csv_4000.csv")
test_labels_4000 = pandas.read_csv("../4000_data/y_test_smpl.csv_4000.csv")
# mlp_test_data(clf, "4000", test_4000, test_labels_4000, visualise=False)

# Question 5
print("\nTesting using 9000 moved testing data ....\n")
test_9000 = pandas.read_csv("../9000_data/x_test_gr_smpl.csv_9000.csv")
test_labels_9000 = pandas.read_csv("../9000_data/y_test_smpl.csv_9000.csv")
# mlp_test_data(clf, "9000", test_9000, test_labels_9000, visualise=True)
