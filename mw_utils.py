def openFile(path):
    with open(path, "r") as file:
        data = file.readlines()
    return data
