import numpy as np
import cv2
import imutils
import pytesseract
from PIL.Image import ImageTransformHandler
pytesseract.pytesseract.tesseract_cmd = r'C:\Program Files\Tesseract-OCR\tesseract.exe'

# Read the image file
image = cv2.imread('Car Images/10.jpg')

# Resize the image - change width to 500
image = imutils.resize(image, width=500)

# Display the original image
cv2.imshow("Original Image", image)
cv2.waitKey(0)

# RGB to Gray scale conversion
gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
cv2.imshow("1 - Grayscale Conversion", gray)
cv2.waitKey(0)

# Noise removal with iterative bilateral filter(removes noise while preserving edges)
gray = cv2.bilateralFilter(gray, 11, 17, 17)
cv2.imshow("2 - Bilateral Filter", gray)
cv2.waitKey(0)

# Find Edges of the grayscale image
edged = cv2.Canny(gray, 170, 200)
cv2.imshow("3 - Canny Edges", edged)
cv2.waitKey(0)

# Find contours based on Edges
cnts, new  = cv2.findContours(edged.copy(), cv2.RETR_LIST, cv2.CHAIN_APPROX_SIMPLE)

# Create copy of original image to draw all contours
img1 = image.copy()
cv2.drawContours(img1, cnts, -1, (0,255,0), 3)
cv2.imshow("4- All Contours", img1)
cv2.waitKey(0)

#sort contours based on their area keeping minimum required area as '30' (anything smaller than this will not be considered)
cnts=sorted(cnts, key = cv2.contourArea, reverse = True)[:30]
NumberPlateCnt = None #we currently have no Number plate contour

# Top 30 Contours
img2 = image.copy()
cv2.drawContours(img2, cnts, -1, (0,255,0), 3)
cv2.imshow("5- Top 30 Contours", img2)
cv2.waitKey(0)

# loop over our contours to find the best possible approximate contour of number plate
count = 0
idx =7
for c in cnts:
        peri = cv2.arcLength(c, True)
        approx = cv2.approxPolyDP(c, 0.02 * peri, True)
        # print ("approx = ",approx)
        if len(approx) == 4:  # Select the contour with 4 corners
            NumberPlateCnt = approx #This is our approx Number Plate Contour

            # Crop those contours and store it in Cropped Images folder
            x, y, w, h = cv2.boundingRect(c) #This will find out co-ord for plate
            new_img = gray[y:y + h, x:x + w] #Create new image
            cv2.imwrite('Cropped Images-Text/' + str(idx) + '.png', new_img) #Store new image
            idx+=1

            break

(h, w) = image.shape[:2]
# Drawing the selected contour on the original image
#print(NumberPlateCnt)
cv2.drawContours(image, [NumberPlateCnt], -1, (0,255,0), 3)
cv2.imshow("Final Image With Number Plate Detected", image)
cv2.waitKey(0)

Cropped_img_loc = 'Cropped Images-Text/7.png'
cv2.imshow("Cropped Image ", cv2.imread(Cropped_img_loc))

# Use tesseract to covert image into string
text = pytesseract.image_to_string(Cropped_img_loc, lang='eng')
print("Number is :", text)

cv2.waitKey(0) #Wait for user input before closing the images displayed

 

cascade = cv2.CascadeClassifier("C:/Users/shuch/Downloads/CODE 2(STATE RECO)/License-Plate-Recognition-main/haarcascade_russian_plate_number.xml")
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


def extract_num(img_filename):
    img = cv2.imread(img_filename)
    # Img To Gray
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    nplate = cascade.detectMultiScale(gray, 1.1, 4)
    # crop portion
    for (x, y, w, h) in nplate:
        wT, hT, cT = img.shape
        a, b = (int(0.02 * wT), int(0.02 * hT))
        plate = img[y + a:y + h - a, x + b:x + w - b, :]
        # make the img more darker to identify LPR
        kernel = np.ones((1, 1), np.uint8)
        plate = cv2.dilate(plate, kernel, iterations=1)
        plate = cv2.erode(plate, kernel, iterations=1)
        plate_gray = cv2.cvtColor(plate, cv2.COLOR_BGR2GRAY)
        (thresh, plate) = cv2.threshold(plate_gray, 127, 255, cv2.THRESH_BINARY)
        # read the text on the plate
        read = pytesseract.image_to_string(plate)
        read = ''.join(e for e in read if e.isalnum())
        stat = read[0:2]
        ## states
        try:
            print('Car belongs to',states[stat])
        except:
            print('State not recognised!!')
            print(read)
        cv2.rectangle(img, (x, y), (x + w, y + h), (51, 51, 255), 2)
        cv2.rectangle(img, (x - 1, y - 40), (x + w + 1, y), (51, 51, 255), -1)
        cv2.putText(img, read, (x, y - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.9, (255, 255, 255), 2)

        cv2.imshow("plate", plate)

    cv2.imwrite("Result.png", img)
    cv2.imshow("Result", img)
    if cv2.waitKey(0) == 113:
        exit()
    cv2.destroyAllWindows()


extract_num('Car Images/10.jpg')