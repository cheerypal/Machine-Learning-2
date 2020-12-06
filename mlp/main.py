from sklearn.neural_network import MLPClassifier
from sklearn.model_selection import cross_val_predict
from sklearn.metrics import confusion_matrix, accuracy_score
from sklearn import metrics
import numpy as np
import pandas as pd
import Metrics as mt
import matplotlib.pyplot as plt

# total training data
data = pd.read_csv("../training_data/x_train_gr_smpl.csv")
labels = pd.read_csv("../training_data/y_train_smpl.csv")
testingData = pd.read_csv("../testing_data/x_test_gr_smpl.csv")
testingLabels = pd.read_csv("../testing_data/y_test_smpl.csv")
labels = np.ravel(labels)

print("MLP classifier starting")
# Init classifier
clf = MLPClassifier(max_iter=500, random_state=1)
print("MLP classifier finished")

# Run cross validation on the classifier
label_predictions = cross_val_predict(clf, data, labels, cv=10)
confusion_matrix = confusion_matrix(labels, label_predictions)

# Print confusion matrix using the predicted labels
print(confusion_matrix)

# get precision, recall and f1 measure
print("\n", metrics.classification_report(labels, label_predictions))
auc_arr = []
for i in range(0, 5):
    # gets cross val probability of the class - ROC Metric
    file = np.ravel(pd.read_csv("../training_data/y_train_smpl_" + str(i) + ".csv"))
    print("ROC class: ", i)
    # get ROC values and TPR, FPR for ROC curve
    probs = cross_val_predict(clf, data, file, method="predict_proba", cv=10)
    fpr, tpr, thresholds = metrics.roc_curve(file, probs[:, 1])
    auc_arr.append(metrics.roc_auc_score(file, probs[:, 1]))
    plt.plot(fpr, tpr, label="Class " + str(i))

plt.title("ROC Visualisation")
plt.xlabel("False Positive Rate")
plt.ylabel("True Positive Rate")
plt.legend()
plt.show()

# Get accuracy of the cross validation
accuracy = accuracy_score(labels, label_predictions)
print("\nAccuracy: " + str(accuracy))


# Question 3
# Run and get the results of the testing data used on the classifier
print("\nTesting using dataset testing data ....\n")
mt.classifier_tester(clf, "test", data, labels, testingData, testingLabels, visualise=True)

# Question 4
# Run and get the results of the newly created 4000 instance moved test and training dataset
print("\nTesting using 4000 moved testing data ....\n")
# New dataset files - run moveInstance before running this section
train_4000 = pd.read_csv("../4000_data/x_train_gr_smpl.csv4000.csv")
train_labels_4000 = pd.read_csv("../4000_data/y_train_smpl.csv4000.csv")
test_4000 = pd.read_csv("../4000_data/x_test_gr_smpl.csv_4000.csv")
test_labels_4000 = pd.read_csv("../4000_data/y_test_smpl.csv_4000.csv")
train_labels_4000 = np.ravel(train_labels_4000)
mt.classifier_tester(clf, "4000", train_4000, train_labels_4000, test_4000,
                     test_labels_4000, visualise=True)

# Question 5
# Run and get the results of the newly created 9000 instance moved test and training dataset
print("\nTesting using 9000 moved testing data ....\n")
# New dataset files - run moveInstance before running this section
train_9000 = pd.read_csv("../9000_data/x_train_gr_smpl.csv9000.csv")
train_labels_9000 = pd.read_csv("../9000_data/y_train_smpl.csv9000.csv")
test_9000 = pd.read_csv("../9000_data/x_test_gr_smpl.csv_9000.csv")
test_labels_9000 = pd.read_csv("../9000_data/y_test_smpl.csv_9000.csv")
train_labels_9000 = np.ravel(train_labels_9000)
mt.classifier_tester(clf, "9000", train_9000, train_labels_9000, test_9000,
                     test_labels_9000, visualise=True)




