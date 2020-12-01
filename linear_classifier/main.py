import pandas as pd
import numpy as np
from sklearn.model_selection import cross_val_predict
from sklearn.metrics import confusion_matrix
from sklearn.linear_model import SGDClassifier
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import make_pipeline
from sklearn.metrics import accuracy_score
import Metrics as mt
from sklearn import metrics


# total training data
data = pd.read_csv("../training_data/x_train_gr_smpl.csv")
labels = pd.read_csv("../training_data/y_train_smpl.csv")
testingData = pd.read_csv("../testing_data/x_test_gr_smpl.csv")
testingLabels = pd.read_csv("../testing_data/y_test_smpl.csv")

labels = np.ravel(labels)

# Always scale the input. The most convenient way is to use a pipeline.
# Init classifier
# Run cross validation on the classifier
linear_classifier = make_pipeline(StandardScaler(), SGDClassifier(max_iter=500, tol=1e-3))
label_predictions = cross_val_predict(linear_classifier, data, labels, cv=10)
confusion_matrix = confusion_matrix(labels, label_predictions)

# Print confusion matrix using the predicted labels
print(confusion_matrix)

# get precision, recall and f1 measure
print("\n", metrics.classification_report(labels, label_predictions))

# get tpr, fpr and ROC area
print("\n", mt.get_TPR_FPR(label_predictions, "train", visualise=True))
print("\n", mt.get_ROC_AREA(label_predictions, "train"))

# Get accuracy of the cross validation
accuracy = accuracy_score(labels, label_predictions)
print("\nAccuracy: " + str(accuracy))


# Question 3
# Run and get the results of the testing data used on the classifier
print("\nTesting using dataset testing data ....\n")
mt.classifier_tester(linear_classifier, "test", data, labels, testingData, testingLabels, visualise=False)

# Question 4
# Run and get the results of the newly created 4000 instance moved test and training dataset
print("\nTesting using 4000 moved testing data ....\n")
# New dataset files - run moveInstance before running this section
train_4000 = pd.read_csv("../4000_data/x_train_gr_smpl.csv4000.csv")
train_labels_4000 = pd.read_csv("../4000_data/y_train_smpl.csv4000.csv")
test_4000 = pd.read_csv("../4000_data/x_test_gr_smpl.csv_4000.csv")
test_labels_4000 = pd.read_csv("../4000_data/y_test_smpl.csv_4000.csv")
train_labels_4000 = np.ravel(train_labels_4000)
mt.classifier_tester(linear_classifier, "4000", train_4000, train_labels_4000, test_4000,
                     test_labels_4000, visualise=False)

# Question 5
# Run and get the results of the newly created 9000 instance moved test and training dataset
print("\nTesting using 9000 moved testing data ....\n")
# New dataset files - run moveInstance before running this section
train_9000 = pd.read_csv("../9000_data/x_train_gr_smpl.csv9000.csv")
train_labels_9000 = pd.read_csv("../9000_data/y_train_smpl.csv9000.csv")
test_9000 = pd.read_csv("../9000_data/x_test_gr_smpl.csv_9000.csv")
test_labels_9000 = pd.read_csv("../9000_data/y_test_smpl.csv_9000.csv")
train_labels_9000 = np.ravel(train_labels_9000)
mt.classifier_tester(linear_classifier, "9000", train_9000, train_labels_9000, test_9000,
                     test_labels_9000, visualise=False)



