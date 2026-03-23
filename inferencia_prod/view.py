import cv2


def text_overlay(frame, text: str, org=(20,20)):
    scale = 1.2
    color = (0,0,255)
    cv2.putText(frame, text, org=org, fontFace=cv2.FONT_HERSHEY_DUPLEX, fontScale=scale, color=color)
