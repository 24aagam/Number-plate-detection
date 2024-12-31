import cv2
import os

# Haar Cascade
haarcascade = r"C:\Users\Aagam Shah\Downloads\Experiment\Number plate detection\detection\Lib\site-packages\cv2\data\haarcascade_russian_plate_number.xml"

cap = cv2.VideoCapture(0)

if not cap.isOpened():
    print("Error: Could not access the camera.")
    exit()

# Set the frame width to 640 pixels
cap.set(cv2.CAP_PROP_FRAME_WIDTH, 640)

# Set the frame height to 480 pixels
cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 480)

min_area = 500

count = 0 

while True:

    success, img = cap.read()
    
    if not success:
        print("Error: Failed to capture frame.")
        continue
    
    plate_cascade = cv2.CascadeClassifier(haarcascade)

    img_gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

    plates = plate_cascade.detectMultiScale(img_gray, scaleFactor = 1.1, minNeighbors = 4)

    for (x, y, w, h) in plates:

        area = w*h
        
        # Detection 
        if area>min_area:
        
            cv2.rectangle(img, (x,y), (x+w,y+h), (0,255,0), 2)

            cv2.putText(img, "Number Plate", (x, y-5), cv2.FONT_HERSHEY_COMPLEX_SMALL, 1, (255,0,255), 2)

            img_roi = img[y: y+h, x: x+w]

            cv2.imshow("ROI", img_roi)

    cv2.imshow("Result", img)

    # Save the detected plate on pressing 's'
    if cv2.waitKey(1) & 0xFF == ord('s'):
        os.makedirs("plates", exist_ok=True)  
        file_path = os.path.join("plates", f"scanned_img_{count}.jpg")
        cv2.imwrite(file_path, img_roi)
        cv2.rectangle(img, (0,200), (640,300), (0, 255), cv2.FILLED)
        cv2.putText(img, "Plate Saved", (150, 265), cv2.FONT_HERSHEY_COMPLEX_SMALL,  1, (255, 0, 255), 2)
        cv2.imshow("Result", img)
        cv2.waitKey(500)
        count+=1

