import pandas
import os
import numpy as np


# This randomises the files the same way as the seed is set
np.random.seed(3)


# Moves instances from the bigger files x_train_gr_smpl.csv and y_train_smpl.csv to their testing counterparts
# depending on how many instances are specified.
# Inputs : numberOfInstances
# Outputs : New Training and Testing Files with new numbers of instances
def moveBiggerFiles(numberOfInstances):
    # ################################# Move Data ####################################
    TRAIN_FILE_DATA = "x_train_gr_smpl.csv"  # MOVE DATA FROM THIS FILE
    TEST_FILE_DATA = "x_test_gr_smpl.csv"    # TO THIS FILE

    # Open files
    training_file_data = pandas.read_csv("training_data/" + TRAIN_FILE_DATA)
    # shuffle data so that random data is moved
    np.random.shuffle(training_file_data.values)
    testing_file_data = pandas.read_csv("testing_data/" + TEST_FILE_DATA)

    # Files with new data
    # New Training file containing the remaining number of instances
    training_file_4000 = training_file_data[numberOfInstances-1:]
    # New testing file containing the first number of instances
    testing_file_4000 = testing_file_data.append(training_file_data[:numberOfInstances])

    # output to new files
    training_file_4000.to_csv(str(numberOfInstances)+"_data/" + TRAIN_FILE_DATA + "" + str(numberOfInstances) +
    ".csv", index=False)

    testing_file_4000.to_csv(str(numberOfInstances)+"_data/" + TEST_FILE_DATA + "_" + str(numberOfInstances) +
                             ".csv", index=False)

    # ################################# Move Labels Data ####################################


    TRAIN_FILE_LABELS = "y_train_smpl.csv"  # MOVE LABELS FROM THIS FILE
    TEST_FILE_LABELS = "y_test_smpl.csv"    # TO THIS FILE

    # Open files
    training_file_labels = pandas.read_csv("training_data/" + TRAIN_FILE_LABELS)
    # shuffle data so that random data is moved
    np.random.shuffle(training_file_labels.values)
    testing_file_labels = pandas.read_csv("testing_data/" + TEST_FILE_LABELS)

    # Files with new data
    # New Training file containing the remaining number of instances
    training_file_4000 = training_file_labels[numberOfInstances-1:]
    # New testing file containing the first number of instances
    testing_file_4000 = testing_file_labels.append(training_file_labels[:numberOfInstances])

    # output to new files
    training_file_4000.to_csv(str(numberOfInstances) + "_data/" + TRAIN_FILE_LABELS + "" + str(numberOfInstances) +
                              ".csv", index=False)

    testing_file_4000.to_csv(str(numberOfInstances)+"_data/" + TEST_FILE_LABELS + "_" + str(numberOfInstances) +
                             ".csv", index=False)


# Moves instances from files like y_train_smpl_0.csv to their testing counterparts
# depending on how many instances are specified.
# Inputs : numberOfInstances
# Outputs : New Testing Files with new numbers of instances
def moveBinaryFiles(numberOfInstances):
    for i in range(0, 10):

        # open files
        training_file_labels = pandas.read_csv("training_data/y_train_smpl_"+str(i)+".csv")
        # shuffle data so that random data is moved
        np.random.shuffle(training_file_labels.values)
        testing_file_labels = pandas.read_csv("testing_data/y_test_smpl_"+str(i)+".csv")

        # File with new data
        testing_file_4000 = testing_file_labels.append(training_file_labels[:numberOfInstances])
        # output to new file
        testing_file_4000.to_csv(str(numberOfInstances)+"_data/y_test_smpl_"+str(i)+".csv_" + str(numberOfInstances) +
                                 ".csv", index=False)


# Main method for moving instances
def moveData(numberOfInstances):
    os.makedirs(str(numberOfInstances)+"_data", exist_ok=True)
    moveBiggerFiles(numberOfInstances)
    moveBinaryFiles(numberOfInstances)


moveData(4000)
moveData(9000)


