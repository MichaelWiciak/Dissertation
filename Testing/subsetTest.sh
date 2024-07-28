# point of this file is to test if we create a subset of data, that the size of the new data is equal to percentage provided * size of original data
# so just run it and assert the sizes are equal

# get the size of the original data for each of the 3 datasets, test, train and valid
originalSizeTest=$(wc -l < ../CodeSearchNet/python/test.jsonl)
originalSizeTrain=$(wc -l < ../CodeSearchNet/python/train.jsonl)
originalSizeValid=$(wc -l < ../CodeSearchNet/python/valid.jsonl)

# echo the original sizes
echo "Original size of test data: $originalSizeTest"
echo "Original size of train data: $originalSizeTrain"
echo "Original size of valid data: $originalSizeValid"

# echo that we are testing
echo "Testing subsetGeneration.py"

get_time() {
    echo $(date +"%T")
}

# now run the python script called subsetGeneration.py with command line arguments
# python subsetGeneration.py [number] [train/test/valid]
# where number is the percentage of the original data you want to keep
# and train/test/valid is the dataset you want to create a subset of

# the number for the tests should be 10, 20, 30, 40, 50, 60, 70, 80, 90, 100
# the dataset for the tests should be test, train, valid

# give space
echo ""

# test the test dataset in a for loop
echo "Testing on test dataset"
name="test"
for ((i = 10; i <= 100; i += 10));
do
    # Start timing
    start_time=$(get_time)
    python ../DatasetManipulations/subsetGeneration.py $i test
     # End timing
    end_time=$(get_time)
    # say the start and end time
    echo "Start time: $start_time"
    echo "End time: $end_time"
    # the created subset file is in [name][percentage]/[name]_jsonl
    directory="subset/$name$i/$name.jsonl"
    subsetSize=$(wc -l < $directory)
    if [ $((subsetSize)) -ne $((originalSizeTest * i / 100)) ]
    then
        echo "Test failed for $i percent of test data"
    else
        echo "Test passed for $i percent of test data"
    fi
    # give space
    echo ""
done

# test the train dataset in a for loop
name="train"
echo "Testing on train dataset"
for ((i = 10; i <= 100; i += 10));
do
    # Start timing
    start_time=$(get_time)
    python ../DatasetManipulations/subsetGeneration.py $i test
     # End timing
    end_time=$(get_time)
    # say the start and end time
    echo "Start time: $start_time"
    echo "End time: $end_time"
    # the created subset file is in [name][percentage]/[name]_jsonl
    directory="subset/$name$i/$name.jsonl"
    subsetSize=$(wc -l < $directory)
    if [ $((subsetSize)) -ne $((originalSizeTrain * i / 100)) ]
    then
        echo "Test failed for $i percent of train data"
    else
        echo "Test passed for $i percent of train data"
    fi
    # give space
    echo ""
done

# test the valid dataset in a for loop
name="valid"
echo "Testing on valid dataset"
for ((i = 10; i <= 100; i += 10));
do
    # Start timing
    start_time=$(get_time)
    python ../DatasetManipulations/subsetGeneration.py $i test
     # End timing
    end_time=$(get_time)
    # say the start and end time
    echo "Start time: $start_time"
    echo "End time: $end_time"
    # the created subset file is in [name][percentage]/[name]_jsonl
    directory="subset/$name$i/$name.jsonl"
    subsetSize=$(wc -l < $directory)
    if [ $((subsetSize)) -ne $((originalSizeValid * i / 100)) ]
    then
        echo "Test failed for $i percent of valid data"
    else
        echo "Test passed for $i percent of valid data"
    fi
    # give space
    echo ""
done