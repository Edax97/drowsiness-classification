import cv2

import utils
from object_detector import ObjectDetector
from camera import find_camera

if __name__ == "__main__":
    cap = find_camera()
    object_detector = ObjectDetector(
        model_filename='object_model/model_28_08.tflite',
        allowlist=['cigarro', 'cinturon', 'celular'],
        threshold_telefono=0.25,
        threshold_cigarro=0.35,
        threshold_cinturon=0.5,
    )
    while True:
        ret, frame = cap.read()
        if not ret:
            break
        #---------------------

        result_cigarro, result_cinturon, result_telefono = object_detector.detect_objects(frame)
        status_cigarro, status_cinturon, status_telefono = object_detector.get_status()

        utils.display(frame, f"Telefono result: {result_telefono}", 0)
        utils.display(frame, f"Cigarro result: {result_cigarro}", 1)
        utils.display(frame, f"Cinturon result: {result_cinturon}", 2)

        utils.display(frame, "---", 3)
        utils.display(frame, f"Telefono status: {status_telefono}", 4)
        utils.display(frame, f"Cinturon status: {status_cinturon}", 5)

        #---------------------
        cv2.imshow('test object detection', frame)
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break
    cap.release()
    cv2.destroyAllWindows()