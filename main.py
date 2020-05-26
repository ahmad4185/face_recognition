import ExtractFaceFeatures as ef 
import FeatureVector as fv 
import csvLibrary as cs 
import numpy as np
import pickle
import pandas as pd
import cv2



def Create_File(DATADIR, RESOLUTION):
    data_matrix = []
    head = fv.HeaderGeneration(RESOLUTION*RESOLUTION)
    vector = ef.Face(DATADIR, RESOLUTION)
    vector.append('Unknown')
    data_matrix.append(vector)
    data_matrix.insert(0, head)
    data_matrix = np.array(data_matrix)
    
    csv_filename = 'csvFiles\\Unknown.csv'
    cs.generateCSV(data_matrix, csv_filename)
    
    return csv_filename

def Prediction(DATADIR, Trained_Model_Name):
    with open(Trained_Model_Name,'rb') as f:
        mp=pickle.load(f)
   
        test_data = pd.read_csv(DATADIR)
        y = test_data['Label']
        test_data = test_data.drop('Label',axis=1)

        predictions = mp.predict(test_data)
        class_probabilities = mp.predict_proba(test_data)
    
    return [y[0], predictions[0], max(class_probabilities[0])]

def GenerateResults(results, img_path):
    img = cv2.imread(img_path)
    face_cascade = cv2.CascadeClassifier('haar_cascade\\haarcascade_frontalface_default.xml')
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    faces = face_cascade.detectMultiScale(gray, 1.1, 10)
   
    
    font = cv2.FONT_HERSHEY_SIMPLEX 
    org = (faces[0][0]-20, faces[0][1]-20) 
    fontScale = 1
    color = (0, 255, 0) 
    thickness = 2
    
    
        
        
    if (results[2] <= 0.4):
        text = results[0]
        final_text = 'Unknown'
        color = (0, 0, 255) 
    else:
        text = results[1]
        confidence = (round(results[2],8))*100
        confidence = round(confidence,2)
       
        final_text = text+'  ' +str(confidence)+'%'
    
    for (x, y, w, h) in faces:
        cv2.rectangle(img, (x, y), (x+w, y+h), color, thickness*2)
    
    
   
      
       
    # Using cv2.putText() method 
    img = cv2.putText(img, final_text, org, font, fontScale,  
                     color, thickness, cv2.LINE_AA, False) 
    
    return img

    

def main():
    IMG_NAME = input("Please write the image name without extention e.g. (photo-0231): ?")
    DATADIR =  'Runtime_Input\\'+IMG_NAME+'.jpg'
    RESOLUTION = 30
    
    name = Create_File(DATADIR, RESOLUTION)
    MODEL_NAME = 'models\\Trained'
    
    results = Prediction(name, MODEL_NAME)
    
    img = GenerateResults(results, DATADIR)
    
    img_file = 'Runtime_Output\\'+IMG_NAME+'.jpg'
    cv2.imwrite(img_file, img)
    print("Your output image has been saved in the Runtime_Output folder with same image file name")
    print("Thank you!")
    
    cv2.imshow('Frame',img)
    cv2.waitKey()

    
# main file
main()


