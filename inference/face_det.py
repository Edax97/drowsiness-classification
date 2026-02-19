import cv2 as cv

INDEX = ".venv_train/lib64/python3.11/site-packages/"

face_cascade = cv.CascadeClassifier()
if not face_cascade.load("../haarcascade_frontalface_alt.xml"):
    print('--(!)Error loading face cascade')
    exit(0)

def detect_face(frame: cv.Mat):
    gray = cv.cvtColor(frame, cv.COLOR_BGR2GRAY)
    gray = cv.equalizeHist(gray)
    _, w = gray.shape
    _min, _max = int(w*0.15), int(w*0.99)
    faces = face_cascade.detectMultiScale(gray, scaleFactor=1.1, minNeighbors=1, minSize=(_min, _min), maxSize=(_max, _max))
    if len(faces) == 0:
        return False, (0,0,0,0)
    face_main = faces[0]
    return True, face_main



