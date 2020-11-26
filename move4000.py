import pandas
import os

os.makedirs("4000_data", exist_ok=True)


def moveBiggerFiles():
    # CHANGE THESE VARIABLES TO PICK WHAT FILE YOU WANT TO MOVE DATA FROM AND TO

    # Move Data
    TRAIN_FILE_DATA = "x_train_gr_smpl.csv"  # MOVE DATA FROM THIS FILE
    TEST_FILE_DATA = "x_test_gr_smpl.csv"    # TO THIS FILE

    training_file_data = pandas.read_csv("training_data/" + TRAIN_FILE_DATA)
    testing_file_data = pandas.read_csv("testing_data/" + TEST_FILE_DATA)

    # File with new data
    testing_file_4000 = testing_file_data.append(training_file_data[:4000])

    # output to new file
    testing_file_4000.to_csv("4000_data/" + TEST_FILE_DATA + "_4000.csv", index=False)

    # Move Labels Data
    TRAIN_FILE_LABELS = "y_train_smpl.csv"  # MOVE LABELS FROM THIS FILE
    TEST_FILE_LABELS = "y_test_smpl.csv"    # TO THIS FILE

    training_file_labels = pandas.read_csv("training_data/" + TRAIN_FILE_LABELS)
    testing_file_labels = pandas.read_csv("testing_data/" + TEST_FILE_LABELS)

    # File with new data
    testing_file_4000 = testing_file_labels.append(training_file_labels[:4000])
    # output to new file
    testing_file_4000.to_csv("4000_data/" + TEST_FILE_LABELS + "_4000.csv", index=False)


def moveBinaryFiles():
    for i in range(0, 10):
        training_file_labels = pandas.read_csv("training_data/y_train_smpl_"+str(i)+".csv")
        testing_file_labels = pandas.read_csv("testing_data/y_test_smpl_"+str(i)+".csv")

        # File with new data
        testing_file_4000 = testing_file_labels.append(training_file_labels[:4000])
        # output to new file
        testing_file_4000.to_csv("4000_data/y_test_smpl_"+str(i)+".csv_4000.csv", index=False)


def move4000Data():
    moveBiggerFiles()
    moveBinaryFiles()

