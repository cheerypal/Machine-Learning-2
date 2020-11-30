import os
import random
import pandas

# function that outputs a seed for the randomizer.
def seed():
    return 0.01


def randomize_data():
    """
    Takes in a directory of files and randomises all the files the same way.
    """

    # Checks to make sure if the randomized data already exists.
    if os.path.exists("randomized_data"):
        print("Randomized data already exists.")
    else:
        os.mkdir("randomized_data")

    # Loops through the directory, reads in the file to be randomized, randomizes the content, writes data
    # to a new file in the randomized_data directory.
    for filename in os.listdir("training_data"):
        print("Randomizing " + filename)
        file = open("training_data/" + filename, "r")
        training_data = file.readlines()
        header = training_data[0]
        data = training_data[1:]
        # shuffle used to make sure that the data is shuffled the same way
        random.shuffle(data, seed)
        # re-append the headers of the files
        data.insert(0, header)
        # write to a new file
        with open("randomized_data/" + filename[0:len(filename) - 4] + "_randomized.csv", "w") as target:
            for line in data:
                target.write(line)
