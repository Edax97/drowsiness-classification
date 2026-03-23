import cv2


def find_camera():
    max_indices = 20
    for i in range(max_indices):
        if i < -1:
            continue

        _cap = cv2.VideoCapture(i)
        if _cap.isOpened():
            _ret, _frame = _cap.read()
            if _ret:
                return _cap
            _cap.release()

    raise BlockingIOError(f"No se encontró cámara en rango {max_indices}")
