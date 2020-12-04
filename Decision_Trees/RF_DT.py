import numpy as np
import pandas as pd
import Decision_Trees.TFunctions as tf
import Metrics as mt


# Method run the classifier using cross validation and using the testing data supplied and created
def runRF(rf_model, vis_tree, visualise, mean_std, data, labels, testingData, testingLabels, save):
    # Question 1
    # Run and get the results of the cross validation used on the classifier
    tf.crossValidation(rf_model, data, labels, visualise=visualise, mean_std=mean_std)

    # Question 2
    # Visualise the tree plot
    if vis_tree:
        rf_model.fit(data, labels)
        rf_tree = rf_model.estimators_[0]
        tf.visualiseTree(rf_tree, save=save)

    # Question 3
    # Run and get the results of the testing data used on the classifier
    print("\nTesting using a dataset testing data ....\n")
    mt.classifier_tester(rf_model, "test", data, labels, testingData, testingLabels, visualise=visualise)

    # Question 4
    # Run and get the results of the newly created 4000 instance moved test and training dataset
    print("\nTesting using 4000 moved testing data ....\n")
    # New dataset files - run moveInstance before running this section
    train_4000 = pd.read_csv("../4000_data/x_train_gr_smpl.csv4000.csv")
    train_labels_4000 = pd.read_csv("../4000_data/y_train_smpl.csv4000.csv")
    test_4000 = pd.read_csv("../4000_data/x_test_gr_smpl.csv_4000.csv")
    test_labels_4000 = pd.read_csv("../4000_data/y_test_smpl.csv_4000.csv")
    train_labels_4000 = np.ravel(train_labels_4000)
    mt.classifier_tester(rf_model, "4000", train_4000, train_labels_4000, test_4000, test_labels_4000,
                         visualise=visualise)

    # Question 5
    # Run and get the results of the newly created 9000 instance moved test and training dataset
    print("\nTesting using 9000 moved testing data ....\n")
    # New dataset files - run moveInstance before running this section
    train_9000 = pd.read_csv("../9000_data/x_train_gr_smpl.csv9000.csv")
    train_labels_9000 = pd.read_csv("../9000_data/y_train_smpl.csv9000.csv")
    test_9000 = pd.read_csv("../9000_data/x_test_gr_smpl.csv_9000.csv")
    test_labels_9000 = pd.read_csv("../9000_data/y_test_smpl.csv_9000.csv")
    train_labels_9000 = np.ravel(train_labels_9000)
    mt.classifier_tester(rf_model, "9000", train_9000, train_labels_9000, test_9000, test_labels_9000,
                         visualise=visualise)
