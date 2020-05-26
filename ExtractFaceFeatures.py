import cv2
import numpy as np


    

def Face(imagePath, dimension):
    cascPath = "haar_cascade\\haarcascade_frontalface_default.xml"
    
   
    faceCascade = cv2.CascadeClassifier(cascPath)
    image = cv2.imread(imagePath)
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    
  
    faces = faceCascade.detectMultiScale(gray, 1.1, 4)
    
   
    if (len(faces)>0):
        x = faces[0][0]
        y = faces[0][1]
        w=  faces[0][2]
        h= faces[0][3]
        crop_img = gray[y:y+h, x:x+w]
       
        
        width = dimension
        height = dimension
        dim = (width, height)
        
        lower_reso = cv2.resize(crop_img, dim)
        img_converted = np.array(lower_reso)
        final_image = img_converted.ravel()
        
        return final_image.tolist()
    else:
        return None