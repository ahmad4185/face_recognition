import os
import ExtractFaceFeatures as fet
import random 
import numpy as np



def countTotal(classes, path):
    total = 0
    print(path)
    for category in classes :
        location = os.path.join(path, category)
        total = total + len(os.listdir(location))
    return total

def DataMatrixGeneration(CATEGORIES, DATADIR, IMG_SIZE):

    NoOfFiles = countTotal(CATEGORIES, DATADIR)
    d_matrix = []
    count = 0
    for category in CATEGORIES:
        path = os.path.join(DATADIR, category)
        for img in os.listdir(path):
            label = category
            direction = path+'\\'+img
            features = fet.Face(direction, IMG_SIZE)
            features.append(label)
            d_matrix.append(features)
            print(direction, end = '')
            print("   ", int((count/NoOfFiles)*100),'%')
            count = count+1
            
    return d_matrix

def HeaderGeneration(Vector_Size):
    Header = []
    for i in range(Vector_Size):
        Header.append(('Features')+str(i+1))
    Header.append('Label')
    return Header

            

def DataMatrixPreparation(CATEGORIES, DATADIR, IMG_SIZE, shuffle):
    training_data = DataMatrixGeneration(CATEGORIES, DATADIR, IMG_SIZE)
    if (random == True):
        random.shuffle(training_data)
    Header = HeaderGeneration(IMG_SIZE*IMG_SIZE)
    training_data.insert(0, Header)
    training_data = np.array(training_data)

    return training_data



