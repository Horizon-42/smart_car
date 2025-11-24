import cv2 as cv

def detect(image):
    gray = cv.equalizeHist(cv.cvtColor(image, cv.COLOR_BGR2GRAY))
    faces = classifier.detectMultiScale(gray)
    for (x,y,w,h) in faces:
        image = cv.ellipse(image, (x+w//2, y+h//2), (w//2, h//2), 0, 0, 360, (255, 0, 0), 4)
    cv.imshow('Object Detection', image)

classifier = cv.CascadeClassifier(cv.data.haarcascades + "haarcascade_frontalface_default.xml")    
camera = cv.VideoCapture(0)
while True:
    detect(camera.read()[1])
    if cv.waitKey(10) == 27: break