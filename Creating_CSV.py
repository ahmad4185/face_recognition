
import FeatureVectorCSV as fvc
import random


def RandomizedCategories(Max_Classes, No_of_Classes, Enable_Random):
    Classes = []
    
    for i in range(Max_Classes):
        Classes.append('Person'+str(i+1))
    
    if (Enable_Random == True):
        random.shuffle(Classes)
    
    return Classes[0:No_of_Classes]
        

IMG_SIZE = int(input("Please enter the image resolution you want to use (e.g. 30 for 30x30 resolution)= "))
DATA_TYPE = str(input('What is your dataset type? "Train" or "Test" ? (Case Sensitive): '))
no_of_categories = int(input("Please input the number of face categories you want to enter (Maximum 10): "))
SHUFFLE = int(input('You want to randomize the face categories? (1 for Yes and 0 for No): '))

CATEGORIES = RandomizedCategories(10, no_of_categories, SHUFFLE)
ENABLE_RANDOM = True
DATADIR = "Dataset\\"+DATA_TYPE
CSV_FILE_NAME = 'csvFiles\\'+DATA_TYPE+'.csv'


fvc.generateFeatureVectorCSV(CATEGORIES, DATADIR, IMG_SIZE, ENABLE_RANDOM, CSV_FILE_NAME)




