
import cv2

_MARGIN = 10  # pixels
_ROW_SIZE = 10  # pixels
_FONT_SIZE = 1
_FONT_THICKNESS = 1
_TEXT_COLOR = (0, 0, 255)  # red

def display(frame, text, status_id: int):
    font = cv2.FONT_HERSHEY_SIMPLEX
    font_size = 0.5
    font_color = (0, 255, 0)
    font_tich = 1
    text_pos_y_base = 20
    text_pos_x_base = 200
    cv2.putText(
        frame,
        text,
        (text_pos_x_base, text_pos_y_base+ 20 * status_id),
        font,
        font_size,
        font_color,
        font_tich,
        cv2.LINE_AA,
    )

def visualize(frame, detection, class_id: int):
    x, y, w, h = detection

    id2Color = [
        (0, 255, 0),
        (0, 0, 255),
        (255, 0, 0),
        (0, 255, 255),
        (255, 255, 0),
        (255, 0, 255),
    ]
    color = id2Color[class_id % len(id2Color)]
    cv2.rectangle(frame, (x, y), (x + w, y + h), color, 2)