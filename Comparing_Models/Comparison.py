# Point of this program is to take the trained code, code+comments, code+comments+ast models and the default BERT model
# and compare their performance on the validation set
# and store the results in a file so i can draw diagrams from later 
# and compare the performance of the models


import torch
import sys
sys.path.append("..")
import mw_utils as mw
import json
from transformers import pipeline
from transformers import RobertaForMaskedLM, RobertaTokenizer
import time

# singular test to see what the model predicts
def testModel():
    model, tokeniser = loadModel("Code")
    input_text = "def add(a, b): return a <mask> b"
    predicted_output = getPredictedOutput(model, tokeniser, input_text)
    # Print the predicted token
    print(f"Predicted Token: {predicted_output}")
    # was expecting ,
    print(f"Expected Token: '+'")

# the model is found in ../Models/[model_name]
def loadModel(model_name):
    model_directory = "../Models/" + model_name
    # load using pretrained
    model = RobertaForMaskedLM.from_pretrained(model_directory)
    tokeniser = RobertaTokenizer.from_pretrained(model_directory)

    return model, tokeniser
    



def loadTestData():
    # the validation data is found in ../DatasetManipulations/preprocessed/valid/train_code.jsonl
    path = "../DatasetManipulations/preprocessedJS/test/test_code.jsonl"
    data = mw.openFile(path)
    return data



def loadValidationData():
    # the validation data is found in ../DatasetManipulations/preprocessed/valid/train_code.jsonl
    path = "../DatasetManipulations/preprocessedP/valid/valid_code.jsonl"
    data = mw.openFile(path)
    return data

    
def testModelAgainstValidation():
    # find what the model predicts for the validation data
    # and store the results in a file

    # load the validation data
    validation_data = loadValidationData()
    # load the models
    code_model, code_tokeniser = loadModel("Code")
    
    testCounter = 0

    # for each line in the validation data
    for line in validation_data:
        # get the input text
        input_tokens = line["code_tokens"]
        input_text = ' '.join(input_tokens)
        # get the expected output
        expected_output = line["output"]
        # get the predicted output
        predicted_output = getPredictedOutput(code_model, code_tokeniser, input_text)
        # print what it predicts and what was in the output field
        print(f"Predicted: {predicted_output}", end=" ")
        print(f"Expected: {expected_output}")
        
        if testCounter == 10:
            break

        # increment the counter
        testCounter += 1



def openOutputFile():
    # open the file to store the results
    pass 


def getPredictedOutput(model, tokeniser, input_text):
    # Tokenize the code snippet
    input_ids = tokeniser.encode(input_text, return_tensors="pt")

    # Generate predictions
    with torch.no_grad():
        outputs = model(input_ids)
        predictions = outputs.logits[0, input_ids[0].tolist().index(tokeniser.mask_token_id)].argmax().item()

    # Decode the predicted token
    predicted_token = tokeniser.decode(predictions)


    return predicted_token


# test the model
def testDemo(model_int):
    # find what the model predicts for the validation data
    # and store the results in a file

    
    test_data = loadTestData()
    # load the models

    timePerTokenLength = {}
    
    if model_int == 1:
        # load the model that is in ../Models/Code
        code_model, code_tokeniser = loadModel("../Models/CodeCommentsAST")
        fill_mask = pipeline("fill-mask", model=code_model, tokenizer=code_tokeniser)
    
    name = "CodeCommentsAST"

    language = "JS"

    # open a file to where it will write incorrect predictions, what code it was and what it predicted
    # open the file to store the results,
    # if the file does not exist, it will be created
    f = open(f"Results/ErrorAnalysis_{name}_{language}.jsonl", "w")

    testCounter = 1
    total = 0
    correct_predictions = 0

    outputJsonlObjet = {}
    # for each line in the validation data
    for line in test_data:
        # get the input text
        # Assuming your JSON string is stored in a variable named json_string
        json_data = json.loads(line)
        input_tokens = json_data["code_tokens"]

        input_text = ' '.join(input_tokens)

        # if input_tokens > 512, then skip this example
        t_tokens = code_tokeniser.tokenize(input_text)
        # Calculate the sequence length
        sequence_length = len(t_tokens)
        if sequence_length > 510:
            continue

        # get the expected output

        expected_output = json_data["output"][0]
        # get the predicted output

        # time the prediction
        start = time.time()
        predicted_output = fill_mask(input_text)
        end = time.time()


        predicted = predicted_output[0]['token_str']
        # remove any space from predicted
        predicted = predicted.replace(" ", "")
        # print what it predicts and what was in the output field
        # also print the original input
        # also print the counter as 'code example ...'
        # if counter is a multiple of 100, print the counter
        if testCounter % 100 == 0:
            print(f"{name} example {testCounter} out of {len(test_data)}")
        

        # increment the counter
        testCounter += 1
        total += 1


        # calculate the time per token length
        timeTaken = end - start
        # if the sequence length is not in the dictionary, add it
        if sequence_length not in timePerTokenLength:
            timePerTokenLength[sequence_length] = [timeTaken]
        else:
            timePerTokenLength[sequence_length].append(timeTaken)



        # if the predicted output is the same as the expected output
        if predicted == expected_output:
            correct_predictions += 1
        else:
            # write to the file
            outputJsonlObjet["code"] = input_text
            outputJsonlObjet["predicted"] = predicted
            outputJsonlObjet["expected"] = expected_output
            f.write(json.dumps(outputJsonlObjet))
            f.write("\n")
        
        # stop if we have done 2000 examples
        if testCounter == 1000:
            break
    
    
    

# test how each model performs but on a different dataset
# the dataset is in ../CodeNet/masked_functions.jsonl
def cplusplusDatasetTest():
    # load the dataset
    path = "../CodeNet/masked_functions.jsonl"
    data = mw.openFile(path)
    # convert data to jsonl
    data = [json.loads(line) for line in data]

    # load the models
    code_model, code_tokeniser = loadModel("CodeCommentsAST")
    # code_comments_model, code_comments_tokeniser = loadModel("CodeComments")
    # code_comments_ast_model, code_comments_ast_tokeniser = loadModel("CodeCommentsAST")

    name = "codeCommentsAST"

    # open a file to store the results
    f = open(f"Results/cplusplus_results_{name}.txt", "w")

    timePerTokenLength = {}

    testCounter = 0
    total = 0
    correct_predictions = 0
    # for each line in the dataset
    for line in data:
        # get the input text
        input_text = line["code_string"]
        # get the expected output
        expected_output = line["output"]
        # get the predicted output

        # time the prediction
        start = time.time()
        predicted_output = getPredictedOutput(code_model, code_tokeniser, input_text)
        end = time.time()

        # remove any space from predicted
        predicted_output = predicted_output.replace(" ", "")


        # print what it predicts and what was in the output field
        # print(f"Predicted: {predicted_output}", end=" ")
        # print(f"Expected: {expected_output}")

        # increment the counter
        testCounter += 1
        total += 1

        # calculate the time taken
        timeTaken = end - start
        # if the predicted output is the same as the expected output
        if predicted_output == expected_output:
            correct_predictions += 1
        else:
            # write to the file
            f.write(f"Code: {input_text}\n")
            f.write(f"Predicted: {predicted_output}\n")
            f.write(f"Expected: {expected_output}\n")
            f.write("\n")
        
        # add the time taken to the dictionary
        sequence_length = len(input_text.split())
        if sequence_length not in timePerTokenLength:
            timePerTokenLength[sequence_length] = [timeTaken]
        else:
            timePerTokenLength[sequence_length].append(timeTaken)
        
        # if counter is a multiple of 100, print the counter
        if testCounter % 100 == 0:
            print(f"example {testCounter} out of {len(data)}")
        

    f.write(f"Correct predictions: {correct_predictions} out of {total}")
    # write the average time to predict overall the prediction
    f.write("\n")
    f.write("Time taken per token length\n")
    for key in timePerTokenLength:
        times = timePerTokenLength[key]
        average = sum(times) / len(times)
        f.write(f"Token length: {key} Average time: {average}\n")
    
    f.close()

    # write the overall results to file called results.json
    results = {"correct_predictions": correct_predictions, "total": total}
    with open(f"Results/cplusplus_results_{name}.json", "w") as f:
        json.dump(results, f)



# do the same as testDemo() but for different masking lengths
# the dataset is in ../CodeNet/python/test_toeksnOnly.jsonl
def testDemoDifferentMaskingLengths():
    # load the dataset
    path = "../CodeSearchNet/python/test_tokensOnly.jsonl"
    data = mw.openFile(path)

    # convert data to jsonl
    data = [json.loads(line) for line in data]

    # load the models
    code_model, code_tokeniser = loadModel("CodeCommentsAST")
    # code_comments_model, code_comments_tokeniser = loadModel("CodeComments")
    # code_comments_ast_model, code_comments_ast_tokeniser = loadModel("CodeCommentsAST")

    name = "CodeCommentsAST"

    # open a file to store the results
    f = open(f"Results/{name}_maskingLength.txt", "w")

    # the idea is for each object its about to predict, mask the last n tokens (from 2, to 10)
    # and predict multiple times, one by one, until the last token is predicted
    # if we encounter a prediction that is wrong, stop predcting and count that number as wrong, and all future predictions that have
    # higher masking level for the same input since it will predict the same thing so no need to predict again

    outcomeDict = {}
    # add name of the model
    outcomeDict["model"] = name
    
    counter = 0
    # for each object in the dataset
    for line in data:

        

        if counter > 1000:
            break

        # get the input text
        input_tokens = line["code_tokens"]
        input_text = ' '.join(input_tokens)

        # if input_tokens > 512, then skip this example
        t_tokens = code_tokeniser.tokenize(input_text)
        # Calculate the sequence length
        sequence_length = len(t_tokens)
        if sequence_length > 510:
            continue

        if counter % 100 == 0:
            print(f"example {counter} out of {len(data)}")
        
        counter += 1
        
        # for each masking level
        for i in range(2, 11):
            # add the masking level to the dictionary
            if i not in outcomeDict:
                outcomeDict[i] = {"correct": 0, "total": 0}


            # mask the last i tokens
            masked_input, after = maskLastTokens(input_text, i)
            # get the predicted output
            predicted_output = getPredictedOutput(code_model, code_tokeniser, masked_input)
            # remove any space from predicted
            predicted_output = predicted_output.replace(" ", "")
            # get the expected output
            expected_output = after[0]
            # if the predicted output is the same as the expected output
            if predicted_output == expected_output:
                outcomeDict[i]["correct"] += 1
            # increment the total
            outcomeDict[i]["total"] += 1

            # if the prediction is wrong, stop predicting
            if predicted_output != expected_output:
                break
        
    
    # write the results to the file
    for key in outcomeDict:
        if key == "model":
            f.write(f"Model: {outcomeDict[key]}\n")
            continue
        f.write(f"Masking level: {key}\n")
        f.write(f"Correct predictions: {outcomeDict[key]['correct']} out of {outcomeDict[key]['total']}\n")
        f.write("\n")
    
    f.close()

    # write the overall results to file called results.json
    with open(f"Results/{name}_maskingLength.json", "w") as f:
        json.dump(outcomeDict, f)
    


def maskLastTokens(input_text, n):
    # mask the last len(input_text) - 1 - n tokens
    # remove all data after that index
    # save waht you removed in a variable
    after = input_text.split()[-n:]
    # remove the last n tokens
    input_text = input_text.split()[:-n]
    # join the tokens
    input_text = ' '.join(input_text)
    # add the mask token
    input_text += " <mask>"
    
    return input_text, after


# run the model again on codeCOmments model on test data 
# but measure how long it takes to predict for different docstring lengths
# and whether the length of the docstring affects the prediction
def testModelDifferentDocstringLengths():
    # load the dataset
    test_data = loadTestData()
    # load the models

    timePerTokenLength = {}
    
    code_model, code_tokeniser = loadModel("../Models/CodeComments")
    fill_mask = pipeline("fill-mask", model=code_model, tokenizer=code_tokeniser)
    
    name = "CodeComments"
    # the idea is for each object its about to predict, mask the last n tokens (from 2, to 10)
    # and predict multiple times, one by one, until the last token is predicted
    # if we encounter a prediction that is wrong, stop predcting and count that number as wrong, and all future predictions that have
    # higher masking level for the same input since it will predict the same thing so no need to predict again

    outcomeDict = {}
    # add name of the model
    outcomeDict["model"] = name
    
    # open a file to store the results
    f = open(f"Results/{name}_docstringLength.txt", "w")


    counter = 0
    # for each object in the dataset
    for line in test_data:
        # get the input text
        # Assuming your JSON string is stored in a variable named json_string
        json_data = json.loads(line)
        input_tokens = json_data["code_tokens"]

        input_text = ' '.join(input_tokens)

        # if input_tokens > 512, then skip this example
        t_tokens = code_tokeniser.tokenize(input_text)
        # Calculate the sequence length
        sequence_length = len(t_tokens)
        if sequence_length > 510:
            continue

        # conver tto str
        sequence_length = str(sequence_length)
        # get the expected output

        expected_output = json_data["output"][0]
        # get the predicted output

        # time the prediction
        start = time.time()
        predicted_output = fill_mask(input_text)
        end = time.time()

        # remove any space from predicted
        predicted_output = predicted_output[0]['token_str'].replace(" ", "")



        if counter % 100 == 0:
            print(f"example {counter} out of {len(test_data)}")
        
        # if we reached 1001 examples, stop
        if counter > 1000:
            break

        counter += 1
        
        # append the length of the docstring to the dictionary if its not there
        if sequence_length not in outcomeDict:
            outcomeDict[sequence_length] = {"correct": 0, "total": 0, "timeTaken": []}

        # calculate the time taken
        timeTaken = end - start
        outcomeDict[sequence_length]["timeTaken"].append(timeTaken)

        if predicted_output == expected_output:
            outcomeDict[sequence_length]["correct"] += 1
        # increment the total
        outcomeDict[sequence_length]["total"] += 1
    
    # write the results to the file
    for key in outcomeDict:
        if key == "model":
            f.write(f"Model: {outcomeDict[key]}\n")
            continue
        f.write(f"Docstring length: {key}\n")
        f.write(f"Correct predictions: {outcomeDict[key]['correct']} out of {outcomeDict[key]['total']}\n")
        f.write("\n")
    
    f.close()

    # write the overall results to file called results.json
    with open(f"Results/{name}_docstringLength.json", "w") as f:
        json.dump(outcomeDict, f)
    



def testSingleExample():
    # load data from simpleExample.jsonl
    path = "simpleExample.jsonl"
    data = mw.openFile(path)
    # convert data to jsonl
    data = [json.loads(line) for line in data]

    # load the models
    code_model, code_tokeniser = loadModel("CodeComments")

    name = "CodeComments"

    # open a file to store the results
    f = open(f"Results/{name}_simpleExample.txt", "w")

    # predict the next token in the code snippet that is in the data

    for line in data:
        # get the input text
        input_text = line["code_tokens"]
        # get the expected output
        expected_output = line["output"]
        # get the predicted output
        # time it
        start = time.time()
        predicted_output = getPredictedOutput(code_model, code_tokeniser, input_text)
        end = time.time()

        # remove any space from predicted
        predicted_output = predicted_output.replace(" ", "")
        # print what it predicts and what was in the output field
        print(f"Predicted: {predicted_output}", end=" ")
        print(f"Expected: {expected_output}")
        # write to the file
        f.write(f"Code: {input_text}\n")
        f.write(f"Predicted: {predicted_output}\n")
        f.write(f"Expected: {expected_output}\n")
        # save time
        f.write(f"Time taken: {end - start}\n")
        f.write("\n")


    f.close()


def test():
    testDemo(1)
    # testModel()
    # cplusplusDatasetTest()
    # print(maskLastTokens("def add(a, b): return a + b", 3))
    # testDemoDifferentMaskingLengths()
    # testModelDifferentDocstringLengths()
    # testSingleExample()

if __name__ == "__main__":
    # main()
    test()

