import pandas
import os

os.makedirs("9000_data", exist_ok=True)


#CHANGE THESE VARIABLES TO PICK WHAT FILE YOU WANT TO MOVE DATA FROM AND TO
TRAIN_FILE_NAME = "x_train_gr_smpl.csv"  #MOVE DATA FROM THIS FILE
TEST_FILE_NAME = "x_test_gr_smpl.csv"    #TO THIS FILE

training_file = pandas.read_csv("training_data/" + TRAIN_FILE_NAME)
testing_file = pandas.read_csv("testing_data/" + TEST_FILE_NAME)

training_file_9000 = training_file[8999:]
testing_file_9000 = testing_file.append(training_file[:9000])

training_file_9000.to_csv("9000_data/" + TRAIN_FILE_NAME + "_9000.csv", index=False)
testing_file_9000.to_csv("9000_data/" + TEST_FILE_NAME + "_9000.csv", index=False)
