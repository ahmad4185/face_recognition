import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import classification_report,confusion_matrix
from sklearn.svm import SVC
import pickle
import numpy as np



def SVM_Training(DATADIR, Trained_Model_Name):
    

    #Splitting into Data Matrix and Label Classes Vectors
    X = pd.read_csv(DATADIR)
    y = X['Label']
    X = X.drop('Label',axis=1)

    #Spliting the Dataset into train and validation
    X_train, X_test, y_train, y_test =(train_test_split(X, y, train_size=0.70, test_size=0.30))
    print("Dataset has been splitted into 70% Training and 30% Validation")



    svm_model = SVC(kernel='linear', probability = True)
    print("SVM Model is being trained")

    cls =svm_model.fit(X_train,y_train)
    val_predictions=cls.predict(X_test)

    print("The confusion matrix of validation dataset:\n\n")
    print(confusion_matrix(y_test,val_predictions))

    print("The classification report of validation dataset:\n\n")
    print(classification_report(y_test,val_predictions))


    with open(Trained_Model_Name,'wb') as f:
       pickle.dump(svm_model,f)

    print('The model has been saved with with following path')
    print(Trained_Model_Name)


def Test_Accuracy(DATADIR, Trained_Model_Name):
    with open(Trained_Model_Name,'rb') as f:
        mp=pickle.load(f)
   
        test_data = pd.read_csv(DATADIR)
        y = test_data['Label']
        test_data = test_data.drop('Label',axis=1)

        predictions = mp.predict(test_data)
        class_probabilities = mp.predict_proba(test_data)

    print("The confusion matrix of test dataset:\n\n")
    print(confusion_matrix(y,predictions))

    print("The classification report of test dataset:\n\n")
    print(classification_report(y,predictions))

    query = input('Do you want to see the Probability of each test class? ("Yes or No") ')
    query = query.lower()

    if (query[0] == 'y'):

        print('\n\n\n The class probabilities are given below')
        print('Class Label','Predicted Label','Probabilities')
        for i in range(len(class_probabilities)):
            print(y[i], '  ', predictions[i],'    ----',np.round(max(class_probabilities[i])*100,2),'%')
    elif (query[i] == 'n'):
        print("Thank you!")
    else:
        print("Given answer is incorrect")


query = input('Do you want to train the model ("Yes or No") ')
query = query.lower()

if (query[0] == 'y'):
    Model_Name = input('Enter the model name: ')
    File = 'Train'
    DATADIR = 'csvFiles\\'+File+'.csv'
    Trained_Model_File = 'models'+'\\'+Model_Name
    SVM_Training(DATADIR, Trained_Model_File)

query = input('Do you want to see the accuracy results with unseen test dataset ("Yes or No") ')
query = query.lower()
if (query[0] == 'y'):
    Model_Name = input('Enter the model name: ')
    File = 'Test'
    DATADIR = 'csvFiles\\'+File+'.csv'
    Trained_Model_File = 'models'+'\\'+Model_Name
    Test_Accuracy(DATADIR, Trained_Model_File)



   


