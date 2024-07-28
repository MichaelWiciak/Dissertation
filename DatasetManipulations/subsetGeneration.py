# First I need to allow the user to specify a number when trying to call this script

import sys
import os
import json
import random

# This script take a number and a string of either "train" or "test" or "valid" and creates a subset of the dataset
# with the number being the % of the original dataset to be used
# it will first create a new directory called "subset" and then create a new directory inside of that called "train" or "test" or "valid" + the number

# This script will be called from the command line with the following command:
# python subsetGeneration.py [number] [train/test/valid]

# First create the main function that will be called when file is run

def main():
    # first check that the arguments are valid
    checkArgs()
    # all arguments are valid so now assign them to variables
    percent = sys.argv[1]
    dataset = sys.argv[2]
    output_dir = dataset + percent
    # none of the directories exist so now create them
    # create a directory called subset
    if not os.path.exists("subset"):
        os.mkdir("subset")
    # now create a directory called subset/train or subset/test or subset/valid
    if not os.path.exists("subset/" + output_dir):
        os.mkdir(os.path.join("subset", output_dir))
    # now create the new dataset
    createNewDataset(percent, dataset, output_dir)
    # now print a message to the user
    print("New dataset created in subset/" + output_dir + "/" + dataset + ".jsonl")

# now create a function that will check the command line arguments and make sure they are valid


def checkArgs():
    # first check that there are 3 arguments
    check3Arguments()
    # now check that the first argument is a number
    checkNumber()
    # now check that the second argument is either train, test, or valid
    checkString()
    

def check3Arguments():
    if len(sys.argv) != 3:
        print("ERROR: Incorrect number of arguments")
        print("USAGE: python subsetGeneration.py [number] [train/test/valid]")
        sys.exit(1)


def checkNumber():
    try:
        float(sys.argv[1])
    except ValueError:
        print("ERROR: First argument must be a number")
        print("USAGE: python subsetGeneration.py [number] [train/test/valid]")
        sys.exit(1)

def checkString():

    if sys.argv[2] != "train" and sys.argv[2] != "test" and sys.argv[2] != "valid":
        print("ERROR: Second argument must be either train, test, or valid")
        print("USAGE: python subsetGeneration.py [number] [train/test/valid]")
        sys.exit(1)


def createNewDataset(percent, dataset, newDirPath):
    # first I need to open the original dataset
    # which is called either test.jsonl, train.jsonl, or valid.jsonl
    # the path is ../CodeSearchNet/python/
    # first create the path
    input_path = os.path.join("..", "CodeSearchNet", "python", f"{dataset}.jsonl")
    output_path = os.path.join("subset", newDirPath, f"{dataset}.jsonl")
    
    # now open the file jsonl file
    with open(input_path, "r") as jsonl_file, open(output_path, "w") as output_file:
        # first read all the lines of input file
        lines = jsonl_file.readlines()
        # now calculate the subset size
        subset_size = int(len(lines) * (float(percent)/100))
        # now create a random subset of that data
        subset = random.sample(lines, subset_size)
        # write that subset to the output file
        output_file.writelines(subset)
        # now close both files
        jsonl_file.close()
        output_file.close()
    
    print(f"Subset created: {output_path}")

        
# now call the main function if this file was called
if __name__ == "__main__":
    main()