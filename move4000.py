import pandas
import os

os.makedirs("4000_data", exist_ok=True)

#CHANGE THESE VARIABLES TO PICK WHAT FILE YOU WANT TO MOVE DATA FROM AND TO
TRAIN_FILE_NAME = "y_train_smpl_0.csv"  #MOVE DATA FROM THIS FILE
TEST_FILE_NAME = "y_test_smpl_0.csv"    #TO THIS FILE

training_file = pandas.read_csv("training_data/" + TRAIN_FILE_NAME)
testing_file = pandas.read_csv("testing_data/" + TEST_FILE_NAME)


training_file_4000 = training_file[3999:]
testing_file_4000 = testing_file.append(training_file[:4000])

training_file_4000.to_csv("4000_data/" + TRAIN_FILE_NAME + "_4000.csv", index=False)
testing_file_4000.to_csv("4000_data/" + TEST_FILE_NAME + "_4000.csv", index=False)
