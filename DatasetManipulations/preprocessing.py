# Idea behind this code is to take the subset of data
# Then remove strip anything uneccessary from the data
# and create output column by copying the code
# separete the data into the code, code+comments, code+comments+ast datasets

#remove repo, path, func_name, language, code,  sha, url,partition, docstring, code, original_string
# important: code_tokens, docstring_tokens
# comments = docstring??/ i guess, no actual comments there. 


# so for input output dataset, i need to chhose a random token and make it the output? 
# so i replace that token in input with <mask> and then use the output token to train the model?

import sys
import os
import json
import random
import tokenize
from io import BytesIO
import ast

import esprima


# Convert AST nodes to JSON-serializable dictionaries
def ast_to_dict(node):
    result = {}
    for key, value in node.items():
        if isinstance(value, dict) and 'type' in value:
            result[key] = ast_to_dict(value)
        elif isinstance(value, list):
            result[key] = [ast_to_dict(item) for item in value]
        elif isinstance(value, esprima.nodes.Node):
            result[key] = ast_to_dict(vars(value))
        else:
            result[key] = value
    return result

def checkArguments(x):
    if len(sys.argv) != x:
        print("ERROR: Incorrect number of arguments")
        print("USAGE: python preprocessing.py [path to dataset]")
        sys.exit(1)

def checkPath():
    if not os.path.exists(sys.argv[1]):
        print("ERROR: Path does not exist")
        print("USAGE: python preprocessing.py [path to dataset]")
        sys.exit(1)

def checkArgs():
    # first check that there are 2 arguments
    checkArguments(2)
    # now check that the first argument is a valid path
    checkPath()
    

def main():
    # arguments should be:
    # 1. path to the dataset
    # we save the file in preprocessed/[original filename]/[original filename]_preprocessed.jsonl

    # first check that the arguments are valid
    checkArgs()

    # all arguments are valid so now assign them to variables
    path = sys.argv[1]

    # now create the datasets
    createCodeOnlyDataset(path)
    createCodeCommentsDataset(path)
    createCodeCommentsASTDataset(path)

    print("Preprocessing complete")


def createCodeOnlyDataset(path):
    # open the file
    data = openFile(path)
    # remove the unnecessary columns
    data = removeUnnecessary(data)

    # print len of data

    # now save the file
    codeOnly = codeOnlyDataset(data)
    
    codeOnly = masking(codeOnly)
    saveFile(codeOnly, path, "code")
    

def createCodeCommentsDataset(path):
    # open the file
    data = openFile(path)
    # remove the unnecessary columns
    data = removeUnnecessary(data)
    # now save the file
    data = masking(data)
    saveFile(data, path, "codeComments")


def createCodeCommentsASTDataset(path):
    # open the file
    data = openFile(path)
    # remove the unnecessary columns
    data = removeUnnecessaryCodeCommentsAST(data)
    # now calculate the AST for each code_tokens
    data = codeCommentsASTOnly(data)
    # masking
    data = masking(data)
    # remove the original_string column
    data = removeUnnecessaryStringColumn(data)
    saveFile(data, path, "codeCommentsAST")

def openFile(path):
    with open(path, "r") as file:
        data = file.readlines()
    return data


def removeUnnecessaryStringColumn(data):
    for i in range(len(data)):
        data[i].pop("code")
    return data


def masking(data):
    # choose a random token from the code_tokens and replace it with <mask>
    # then use the original token as the output
    # but make sure that the token is not a comment
    # comment is a token that starts with \" and ends with \"
    for i in range(len(data)):
        token = random.choice(data[i]["code_tokens"])
        while token.startswith("\"") and token.endswith("\""):
            token = random.choice(data[i]["code_tokens"])
        data[i]["output"] = [token]
        data[i]["code_tokens"][data[i]["code_tokens"].index(token)] = "<mask>"
        
    return data


def removeUnnecessary(data):
    # first check if the code forms a valid AST
    # if not, then remove it
    # then remove the unnecessary columns

    for i in range(len(data)):
        data[i] = json.loads(data[i])
        data[i]["code"] = checkData(data[i]["code"], data[i]["language"])
    
    # now we are left with None original_string s so we need to remove them

    # print the length of the data
    data = [x for x in data if x["code"] is not None]

    # from the data, remove the following columns:
    # repo, path, func_name, language, code,  sha, url,partition, docstring, code, original_string
    # copy code_tokens to output column, but keep the original code_tokens column
    # that's it
    for i in range(len(data)):
        # data[i] = json.loads(data[i])
        data[i].pop("repo")
        data[i].pop("path")
        data[i].pop("func_name")
        data[i].pop("language")
        data[i].pop("code")
        data[i].pop("sha")
        data[i].pop("url")
        data[i].pop("partition")
        data[i].pop("docstring")
        data[i].pop("original_string")
    
    return data

def checkData(original_string, language, write=False):
    with open("invalid_data.txt", "a") as file:

        # if it forms a valid ast, then it's good
        # if not, then remove it


        if language == "python":
            try:
                ast.parse(original_string)
            except Exception as e:
                # Save the invalid data into a txt file
                if write:
                    file.write(original_string)
                    # Write separator to separate the invalid data entries
                    file.write("\n")
                    file.write("----------------------------------------------------")
                    file.write("\n")
                original_string = None
        elif language == "javascript":
            try:
                esprima.parseScript(original_string)
            except Exception as e:
                # Save the invalid data into a txt file
                if write:
                    file.write(original_string)
                    # Write separator to separate the invalid data entries
                    file.write("\n")
                    file.write("----------------------------------------------------")
                    file.write("\n")
                original_string = None
        else:
            print("Invalid language")
            sys.exit(1)
    closeFile(file)
    
    return original_string


def removeUnnecessaryCodeCommentsAST(data):
     # first check if the code forms a valid AST
    # if not, then remove it
    # then remove the unnecessary columns

    for i in range(len(data)):
        data[i] = json.loads(data[i])
        data[i]["code"] = checkData(data[i]["code"], data[i]["language"], True)
    
    # now we are left with None original_string s so we need to remove them
    data = [x for x in data if x["code"] is not None]

    # remove same as before but dont remove the original_string column
    for i in range(len(data)):
        # data[i] = json.loads(data[i])
        data[i].pop("repo")
        data[i].pop("path")
        data[i].pop("func_name")
        data[i].pop("language")
        data[i].pop("original_string")
        data[i].pop("sha")
        data[i].pop("url")
        data[i].pop("partition")
        data[i].pop("docstring")

    return data

def saveFile(data, path, folderName):
    # check if the directory exists, if not create it
    directory = path.split("/")
    directory = directory[-1].split(".")
    directory = directory[0]
    if not os.path.exists("preprocessedJS/" + directory):
        os.makedirs("preprocessedJS/" + directory)
    # now save the file, but need to put it into folders of based on folderName
    with open("preprocessedJS/" + directory + "/" + directory + "_" + folderName + ".jsonl", "w") as file:
        for i in range(len(data)):
            file.write(json.dumps(data[i]) + "\n")

    closeFile(file)
    
    
def closeFile(file):
    file.close()



def codeOnlyDataset(data):
    # remove docstring_tokens from the file just created
    for i in range(len(data)):
        data[i].pop("docstring_tokens")
        # for each code_tokens, we need to remove every token that is a comment
        # comments are a token that starts with \" and ends with \"
        # Idea: code_tokens = [token for token in code_tokens if not (token.startswith("\"") and token.endswith("\""))]
        data[i]["code_tokens"] = [token for token in data[i]["code_tokens"] if not (token.startswith("\"") and token.endswith("\""))]
    
    return data

def codeCommentsASTOnly(data):
    # First remove the docstring_tokens column
    # Then remove the comments from the code_tokens
    for i in range(len(data)):
        data[i].pop("docstring_tokens")
        data[i]["code_tokens"] = [token for token in data[i]["code_tokens"] if not (token.startswith("\"") and token.endswith("\""))]

    # Now calculate the AST for each code_tokens and append it to the data
    for i in range(len(data)):
        data[i]["ast"] = calculateAST(data[i]["code"], language="javascript")

    return data
    
    
# Given the code tokens, calculate the abstract syntax tree
def calculateAST(original_string, language="python"):
    # the code tokens are in a form of a list
    # code = remove_comments_and_docstrings(original_string)
    code = original_string
    # now calculate the AST
    if language == "python":
        try:
            tree = ast.parse(code)
            tree = ast.dump(tree)

        except Exception as e:
            tree = None
            print("Error calculating AST for code tokens: ", code)
            print(e)
    elif language == "javascript":
        try:
            tree = esprima.parseScript(code) 
            tree = ast_to_dict(tree)
            # convert the AST to string
            tree = json.dumps(tree)

        except Exception as e:
            tree = None
            print("Error calculating AST for code tokens: ", code)
            print(e)
        
    # print("The AST is: ", ast.dump(tree))


    return tree

def remove_comments_and_docstrings(code_string):
    # Tokenize the code
    code_tokens = tokenize.tokenize(BytesIO(code_string.encode('utf-8')).readline)
    
    # Filter out comments and docstrings
    filtered_tokens = []
    in_docstring = False
    for token in code_tokens:
        if token.type == tokenize.STRING:
            in_docstring = not in_docstring
        elif not in_docstring and token.type != tokenize.COMMENT:
            filtered_tokens.append(token)
    
    # Join the remaining tokens to reconstruct the code without comments and docstrings
    cleaned_code = ''.join([token.string for token in filtered_tokens])

    return cleaned_code


    # add two things, the average time taken for a query, 

if __name__ == "__main__":
    main()
