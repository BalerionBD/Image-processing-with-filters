import cv2
import numpy as np

frameWidth = 640    #Frame Width
franeHeight = 480   # Frame Height

plateCascade = cv2.CascadeClassifier("C:/Users/shuch/Downloads/CODE 5 (Comibination of state and real time)/haarcascade_russian_plate_number.xml")
minArea = 500

cap =cv2.VideoCapture(0)
cap.set(3,frameWidth)
cap.set(4,franeHeight)
cap.set(10,150)
count = 0

while True:
    success , img  = cap.read()

    imgGray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

    numberPlates = plateCascade .detectMultiScale(imgGray, 1.1, 4)

    for (x, y, w, h) in numberPlates:
        area = w*h
        if area > minArea:
            cv2.rectangle(img, (x, y), (x + w, y + h), (255, 0, 0), 2)
            cv2.putText(img,"NumberPlate",(x,y-5),cv2.FONT_HERSHEY_COMPLEX,1,(0,0,255),2)
            imgRoi = img[y:y+h,x:x+w]
            cv2.imshow("ROI",imgRoi)
    cv2.imshow("Result",img)
    if cv2.waitKey(1) & 0xFF ==ord('s'):
        cv2.imwrite("C:/Users/shuch/Downloads/CODE 5 (Comibination of state and real time)/Storage"+str(count)+".jpg",imgRoi)
        cv2.rectangle(img,(0,200),(640,300),(0,255,0),cv2.FILLED)
        cv2.putText(img,"Scan Saved",(15,265),cv2.FONT_HERSHEY_COMPLEX,2,(0,0,255),2)
        cv2.imshow("Result",img)
        cv2.waitKey(500)
        count+=1
        states = {"AN": "Andaman and Nicobar",
          "AP": "Andhra Pradesh", "AR": "Arunachal Pradesh",
          "AS": "Assam", "BR": "Bihar", "CH": "Chandigarh",
          "DN": "Dadra and Nagar Haveli", "DD": "Daman and Diu",
          "DL": "Delhi", "GA": "Goa", "GJ": "Gujarat",
          "HR": "Haryana", "HP": "Himachal Pradesh",
          "JK": "Jammu and Kashmir", "KA": "Karnataka", "KL": "Kerala",
          "LD": "Lakshadweep", "MP": "Madhya Pradesh", "MH": "Maharashtra", "MN": "Manipur",
          "ML": "Meghalaya", "MZ": "Mizoram", "NL": "Nagaland", "OD": "Odissa",
          "PY": "Pondicherry", "PN": "Punjab", "RJ": "Rajasthan", "SK": "Sikkim", "TN": "TamilNadu",
          "TR": "Tripura", "UP": "Uttar Pradesh", "WB": "West Bengal", "CG": "Chhattisgarh",
          "TS": "Telangana", "JH": "Jharkhand", "UK": "Uttarakhand"}
## states
        try:
            print('Car belongs to',states[stat])
        except:
            print('State not recognised!!')
