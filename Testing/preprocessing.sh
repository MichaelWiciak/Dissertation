# idea is to check the preprocessed folder and check all the 
# test/train/valid folders and all the datasets inside of them, 
# whether in their input field for each object, it contains at a <mask> token or not.

# directory of the preprocessed folder is
# ../DatasetManipulations/preprocessed/[test/train/valid]/[test/train/valid]_code.jsonl
# ../DatasetManipulations/preprocessed/[test/train/valid]/[test/train/valid]_codeComments.jsonl
# ../DatasetManipulations/preprocessed/[test/train/valid]/[test/train/valid]_codeCommentsAST.jsonl

# echo the start of the test
echo "Start of the test, looking for <mask> in each jsonl"

echo "Testing on test dataset"
name="test"

# first go to the preprocessed folder 
cd ../DatasetManipulations/preprocessed/test

# check the test_code.jsonl file. If it contains <mask> then it would be a success
# note that .jsonl files contains a json object per line so the test is successful if all the lines contain <mask>
if grep -q "<mask>" test_code.jsonl; then
    echo "Test passed for test_code.jsonl, it contains <mask>"
else
    echo "Test failed for test_code.jsonl"
fi

# check the test_codeComments.jsonl file. If it contains <mask> then it would be a success
if grep -q "<mask>" test_codeComments.jsonl; then
    echo "Test passed for test_codeComments.jsonl, it contains <mask>"
else
    echo "Test failed for test_codeComments.jsonl"
fi

# check the test_codeCommentsAST.jsonl file. If it contains <mask> then it would be a success
if grep -q "<mask>" test_codeCommentsAST.jsonl; then
    echo "Test passed for test_codeCommentsAST.jsonl, it contains <mask>"
else
    echo "Test failed for test_codeCommentsAST.jsonl"
fi


echo "Testing on train dataset"
name="train"

# first go to the preprocessed folder
cd ../train

# check the train_code.jsonl file. If it contains <mask> then it would be a success
if grep -q "<mask>" train_code.jsonl; then
    echo "Test passed for train_code.jsonl, it contains <mask>"
else
    echo "Test failed for train_code.jsonl"
fi

# check the train_codeComments.jsonl file. If it contains <mask> then it would be a success
if grep -q "<mask>" train_codeComments.jsonl; then
    echo "Test passed for train_codeComments.jsonl, it contains <mask>"
else
    echo "Test failed for train_codeComments.jsonl"
fi

# check the train_codeCommentsAST.jsonl file. If it contains <mask> then it would be a success
if grep -q "<mask>" train_codeCommentsAST.jsonl; then
    echo "Test passed for train_codeCommentsAST.jsonl, it contains <mask>"
else
    echo "Test failed for train_codeCommentsAST.jsonl"
fi

echo "Testing on valid dataset"
name="valid"

# first go to the preprocessed folder
cd ../valid

# check the valid_code.jsonl file. If it contains <mask> then it would be a success
if grep -q "<mask>" valid_code.jsonl; then
    echo "Test passed for valid_code.jsonl, it contains <mask>"
else
    echo "Test failed for valid_code.jsonl"
fi

# check the valid_codeComments.jsonl file. If it contains <mask> then it would be a success
if grep -q "<mask>" valid_codeComments.jsonl; then
    echo "Test passed for valid_codeComments.jsonl, it contains <mask>"
else
    echo "Test failed for valid_codeComments.jsonl"
fi

# check the valid_codeCommentsAST.jsonl file. If it contains <mask> then it would be a success
if grep -q "<mask>" valid_codeCommentsAST.jsonl; then
    echo "Test passed for valid_codeCommentsAST.jsonl, it contains <mask>"
else
    echo "Test failed for valid_codeCommentsAST.jsonl"
fi



# give space
echo ""

# echo the end of the test
echo "End of the test"

