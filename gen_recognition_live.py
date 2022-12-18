from tensorflow.keras.preprocessing.image import img_to_array
from tensorflow.keras.models import load_model
import numpy as np
import cv2
import os
import cvlib as cv
                    
# loading 3 models
model1 = load_model('gender_detectionCNN1.h5')
model2 = load_model('gender_detectionCNN2.h5')
model3 = load_model('gender_detectionSVM.h5')

# open webcam
webcam = cv2.VideoCapture(0)
    
classes = ['Male','Female']

# loop through frames
while webcam.isOpened():

    # read frame from webcam 
    status, frame = webcam.read()

    # apply face detection
    face, confidence = cv.detect_face(frame)


    # loop through detected faces
    for idx, f in enumerate(face):

        # get corner points of face rectangle        
        (startX, startY) = f[0], f[1]
        (endX, endY) = f[2], f[3]

        # draw rectangle over face
        cv2.rectangle(frame, (startX,startY), (endX,endY), (255,0,0), 2)

        # crop the detected face region
        face_crop = np.copy(frame[startY:endY,startX:endX])

        if (face_crop.shape[0]) < 10 or (face_crop.shape[1]) < 10:
            continue

        # preprocessing for gender detection model
        face_crop = cv2.resize(face_crop, (96,96))
        face_crop = face_crop.astype("float") / 255.0
        face_crop = img_to_array(face_crop)
        face_crop = np.expand_dims(face_crop, axis=0)

        # apply gender detection on face
        ynew1 = model1.predict(face_crop)
        ynew2 = model2.predict(face_crop)
        ynew3 = model3.predict(face_crop)

        ytest1=[]
        ytest2=[]
        ytest3=[]
        for i in ynew1:
            if i[0]>=i[1]:
                ytest1.append([0])
            else:
                ytest1.append([1])

        for i in ynew2:
            if i[0]>=i[1]:
                ytest2.append([0])
            else:
                ytest2.append([1])

        for i in ynew3:
            if i[0]>=i[1]:
                ytest3.append([0])
            else:
                ytest3.append([1])

        for i in range(len(ytest1)):
            cnt_male=0
            cnt_female=0
            if ytest1[i][0]==0:
                cnt_male+=1
            else:
                cnt_female+=1
            if ytest2[i][0]==0:
                cnt_male+=1
            else:
                cnt_female+=1
            if ytest3[i][0]==0:
                cnt_male+=1
            else:
                cnt_female+=1
            if cnt_male>cnt_female:
                idx=0
            else:
                idx=1
        if cnt_male>cnt_female:
            idx=0
        else:
            idx=1
        color=0
        if idx==1:
            color=(180,105,255)
        else:
            color=(255, 0, 0)
        label = classes[idx]

        label = "Prediction: {}".format(label)
# ----------------------------------------------------------------------------
        Y = startY - 10 if startY - 10 > 10 else startY + 10

        # write label and confidence above face rectangle
        cv2.putText(frame, label, (startX, Y),  cv2.FONT_HERSHEY_SIMPLEX,
                    0.7, color, 2)

    # display output
    cv2.imshow("gender detection", frame)

    # press "Q" to stop
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

# release resources
webcam.release()
cv2.destroyAllWindows()