import cv2 as cv
import time
WIDTH = 900
HEIGHT = 540
def view_cam(name: str, camera, process_fn = None):
    cap = cv.VideoCapture(camera)
    if not cap.isOpened():
        print("Error: could not open ", camera)
        return
    while True:
        ok, frame = cap.read()
        if not ok:
            print("Error: could not grab frame")
            break

        # h, w, _ = frame.shape
        frame = cv.resize(frame, (WIDTH, HEIGHT), None)
        frame = cv.cvtColor(frame, cv.COLOR_BGR2GRAY)
        frame = cv.cvtColor(frame, cv.COLOR_GRAY2BGR)
        if process_fn:
            output = process_fn(frame)
        else:
            output = frame
        cv.imshow(name, output)
        if cv.waitKey(100) & 0xFF == ord('q'):
            break

    cap.release()
    cv.destroyAllWindows()

def text_overlay(frame, text: str, org=(20,20)):
    scale = 1.2
    color = (0,0,255)
    cv.putText(frame, text, org=org, fontFace=cv.FONT_HERSHEY_DUPLEX, fontScale=scale, color=color)
