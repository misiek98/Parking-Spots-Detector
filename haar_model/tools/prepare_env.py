import os

listOfFolders = [os.path.join(os.getcwd(), 'haar_model', 'data'),
                 os.path.join(os.getcwd(), 'haar_model', 'neg'),
                 os.path.join(os.getcwd(), 'haar_model', 'pos'),
                 os.path.join(os.getcwd(), 'haar_model', 'results')]

for dir in listOfFolders:
    if not os.path.exists(dir):
        os.mkdir(dir)
