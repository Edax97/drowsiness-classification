import cv2
import time
import mediapipe as mp
import numpy as np
import time
import argparse
import sys
from mediapipe.tasks import python
from mediapipe.tasks.python import vision
from utils import visualize
from PIL import Image
import time
from datetime import datetime, timedelta
from tflite_support.task import core
from tflite_support.task import processor
from tflite_support.task import vision
import utils
import serial
import time
from spidev import SpiDev
from pynput import keyboard
import time
import json
#import subprocess
import face_recognition
import pickle

window_name = r'Detector_Fatiga'
capturing = False
data = ""
data_validada = ""

with open('settings.json', 'r') as json_file:
    parameters = json.load(json_file)

#*>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>
#*          CODIGO LECTOR
def on_press(key):
    global capturing, data,data_validada
    try:
        # Verifica si es un carácter imprimible
        if key.char:
            data += key.char
            capturing = True
    except AttributeError:
        # Captura teclas especiales, como Enter
        if key == keyboard.Key.enter and capturing:
            # print(f"Data capturada: {data}")
            data_validada = data
            data = ""
            capturing = False

def on_release(key):
    if key == keyboard.Key.esc:
        # Detiene la escucha cuando se presiona Esc
        return False

listener = keyboard.Listener(on_press=on_press, on_release=on_release).start()
#*>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>


def find_working_camera(max_indices=30):
    for index in range(max_indices):
        cap = cv2.VideoCapture(index)
        print("Indice:",index) 
        if cap.isOpened():
            ret, frame = cap.read()
            if ret:
                print(f"Cámara encontrada en el índice {index}")
                return cap
            cap.release()
    raise ValueError("No se encontró ninguna cámara disponible")

#model = 'efficientdet_lite0.tflite'
#model = 'best_16_true_100.tflite'
#model = 'best_cinturon.tflite'
model = parameters["model_filename"]

num_threads = 4
WIDTH = parameters["width"]
HEIGHT = parameters["height"]

x = 0
y = 0
z = 0
HEAD_DETECTION_RANGE = parameters["head_detection_range"]
FACE_ID_MODEL = parameters["face_id_model"]

cam = find_working_camera()
# cam.set(cv2.CAP_PROP_FRAME_WIDTH, WIDTH)
# cam.set(cv2.CAP_PROP_FRAME_HEIGHT, HEIGHT)
cam.set(cv2.CAP_PROP_FPS, 30)

pos = (20, 60)
font = cv2.FONT_HERSHEY_SIMPLEX
height = 1.5
weight = 3
myColor = (255, 0, 0)

fps = 0

# OBJECT DETECTION VARIABLES
status_cellphone = False
status_cigarrete = False
status_distraccion = False
status_glasses = True
status_cinturon = True
status_ear = 0

# DISTRACTION PARAMETERS
status_head = ""
status_mar = 0.0

# EAR SETTINGS
# The chosen 12 points:   P1,  P2,  P3,  P4,  P5,  P6
chosen_left_eye_idxs = [362, 385, 387, 263, 373, 380]
chosen_right_eye_idxs = [33, 160, 158, 133, 153, 144]
all_chosen_idxs = chosen_left_eye_idxs + chosen_right_eye_idxs
current_status_eyes = ""

# MAR SETTINGS
MOUTH_INDEXES = [61, 81, 311, 291, 308, 402, 14, 178]

# FACE ID
process_this_frame = True
face_id_contador = 0
persona_id = ''

# MEDIAPIPE SETTINGS
mp_facemesh = mp.solutions.face_mesh.FaceMesh(False, 1, True, 0.7)
mp_drawing = mp.solutions.drawing_utils
mp_circle = mp_drawing.DrawingSpec(
    thickness=1, circle_radius=1, color=(255, 255, 255))
mp_line = mp_drawing.DrawingSpec(
    thickness=1, circle_radius=1, color=(255, 255, 255))
denormalize_coordinates = mp_drawing._normalized_to_pixel_coordinates

# OBJECT DETECTION SETTINGS
base_options = core.BaseOptions(
    file_name=model, use_coral=False, num_threads=num_threads)
detection_options = processor.DetectionOptions(
    max_results=4, score_threshold=parameters["model_threshold"], category_name_allowlist=parameters["objects"])
options = vision.ObjectDetectorOptions(
    base_options=base_options, detection_options=detection_options)
detector = vision.ObjectDetector.create_from_options(options)
tStart = time.time()

def predict(X_img_path, knn_clf=None, model_path=None, distance_threshold=0.6):
    # Load a trained KNN model (if one was passed in)
    if knn_clf is None:
        with open(model_path, 'rb') as f:
            knn_clf = pickle.load(f)

    # Load image file and find face locations
    X_img = X_img_path
    X_face_locations = face_recognition.face_locations(X_img)

    # If no faces are found in the image, return an empty result.
    if len(X_face_locations) == 0:
        return []

    # Find encodings for faces in the test iamge
    faces_encodings = face_recognition.face_encodings(X_img, known_face_locations=X_face_locations)

    # Use the KNN model to find the best matches for the test face
    closest_distances = knn_clf.kneighbors(faces_encodings, n_neighbors=1)
    are_matches = [closest_distances[0][i][0] <= distance_threshold for i in range(len(X_face_locations))]

    # Predict classes and remove classifications that aren't within the threshold
    return [(pred, loc) if rec else ("unknown", loc) for pred, loc, rec in zip(knn_clf.predict(faces_encodings), X_face_locations, are_matches)]


def distance(point_1, point_2):
    """Calculate l2-norm between two points"""
    dist = sum([(i - j) ** 2 for i, j in zip(point_1, point_2)]) ** 0.5
    return dist

def calculate_MAR(mouth_landmarks):
    A = np.linalg.norm(mouth_landmarks[2] - mouth_landmarks[6])  # Vertical distance
    B = np.linalg.norm(mouth_landmarks[0] - mouth_landmarks[4])  # Horizontal distance
    mar = A / B
    return mar

def get_ear(landmarks, refer_idxs, frame_width, frame_height):
    """
    Calculate Eye Aspect Ratio for one eye.

    Args:
        landmarks: (list) Detected landmarks list
        refer_idxs: (list) Index positions of the chosen landmarks
                            in order P1, P2, P3, P4, P5, P6
        frame_width: (int) WIDTH of captured frame
        frame_height: (int) Height of captured frame

    Returns:
        ear: (float) Eye aspect ratio
    """
    try:
        # Compute the euclidean distance between the horizontal
        coords_points = []
        for i in refer_idxs:
            lm = landmarks[i]
            coord = denormalize_coordinates(
                lm.x, lm.y, frame_width, frame_height)
            coords_points.append(coord)

        # Eye landmark (x, y)-coordinates
        P2_P6 = distance(coords_points[1], coords_points[5])
        P3_P5 = distance(coords_points[2], coords_points[4])
        P1_P4 = distance(coords_points[0], coords_points[3])

        # Compute the eye aspect ratio
        ear = (P2_P6 + P3_P5) / (2.0 * P1_P4)

    except:
        ear = 0.0
        coords_points = None

    return ear, coords_points

def get_head(face_landmark, face_2d, face_3d, img_h, img_w):

    for idx, lm in enumerate(face_landmark):
        if (
            idx == 33
            or idx == 263
            or idx == 1
            or idx == 61
            or idx == 291
            or idx == 199
        ):
            if idx == 1:
                nose_2d = (lm.x * img_w, lm.y * img_h)
                nose_3d = (lm.x * img_w, lm.y * img_h, lm.z * 3000)

            x, y = int(lm.x * img_w), int(lm.y * img_h)
            face_2d.append([x, y])  # Get the 2D Coordinates
            face_3d.append([x, y, lm.z])  # Get the 3D Coordinates

    # Convert it to the NumPy array
    face_2d = np.array(face_2d, dtype=np.float64)
    # Convert it to the NumPy array
    face_3d = np.array(face_3d, dtype=np.float64)

    focal_length = 1 * img_w  # The camera matrix

    cam_matrix = np.array(
        [
            [focal_length, 0, img_h / 2],
            [0, focal_length, img_w / 2],
            [0, 0, 1],
        ]
    )

    # The distortion parameters
    dist_matrix = np.zeros((4, 1), dtype=np.float64)
    success, rot_vec, trans_vec = cv2.solvePnP(
        face_3d, face_2d, cam_matrix, dist_matrix)  # Solve PnP
    rmat, jac = cv2.Rodrigues(rot_vec)  # Get rotational matrix
    angles, mtxR, mtxQ, Qx, Qy, Qz = cv2.RQDecomp3x3(rmat)  # Get angles

    # Get the y rotation degree
    x = angles[0] * 360
    y = angles[1] * 360
    z = angles[2] * 360

    # Display the nose direction
    nose_3d_projection, jacobian = cv2.projectPoints(
        nose_3d, rot_vec, trans_vec, cam_matrix, dist_matrix)

    p1 = (int(nose_2d[0]), int(nose_2d[1]))
    p2 = (int(nose_2d[0] + y * 10), int(nose_2d[1] - x * 10))

    if y < HEAD_DETECTION_RANGE * -1:
        status_head = "IZQ"
        # status_head = -1
    elif y > HEAD_DETECTION_RANGE:
        status_head = "DER"
        # status_head = 1
    elif x < HEAD_DETECTION_RANGE * -1:
        status_head = "ABAJO"
        # status_head = -2
    elif x > HEAD_DETECTION_RANGE:
        status_head = "ARRIBA"
        # status_head = 2
    else:
        status_head = "ENFRENTE"
        # status_head = 0

    return face_2d, face_3d, p1, p2, x, y, z, status_head

def calculate_avg_ear(landmarks, left_eye_idxs, right_eye_idxs, image_w, image_h):
    """Calculate Eye aspect ratio"""
    left_ear, left_lm_coordinates = get_ear(
        landmarks, left_eye_idxs, image_w, image_h)
    right_ear, right_lm_coordinates = get_ear(
        landmarks, right_eye_idxs, image_w, image_h
    )
    Avg_EAR = (left_ear + right_ear) / 2.0

    return Avg_EAR, (left_lm_coordinates, right_lm_coordinates)

def get_glasses(img_org, landmarks, frame_width, frame_height):
    # GLASSES SETTINGS
    glass_id = [55, 285, 174, 399]
    coords_points = []

    for i in glass_id:
        lm = landmarks[i]
        coord = denormalize_coordinates(lm.x, lm.y, frame_width, frame_height)
        coords_points.append(coord)

    x_values = [coord[0] for coord in coords_points]
    y_values = [coord[1] for coord in coords_points]

    # Find the maximum and minimum values for x and y
    max_x = max(x_values)
    min_x = min(x_values)
    max_y = max(y_values)
    min_y = min(y_values)

    img_org = Image.fromarray(img_org.astype('uint8'), 'RGB')
    img2 = img_org.crop((min_x, min_y, max_x, max_y))
    img_blur = cv2.GaussianBlur(np.array(img2), (3, 3), sigmaX=0, sigmaY=0)
    edges = cv2.Canny(image=img_blur, threshold1=100, threshold2=200)
    edges_center = edges.T[(int(len(edges.T)/2))]
    if 255 in edges_center:
        return True
    else:
        return False

# MEDIAPIPE SETTINGS
mp_facemesh = mp.solutions.face_mesh.FaceMesh(False, 1, True, 0.5)
mp_drawing = mp.solutions.drawing_utils
mp_circle = mp_drawing.DrawingSpec(
    thickness=1, circle_radius=1, color=(255, 255, 255))
mp_line = mp_drawing.DrawingSpec(
    thickness=1, circle_radius=1, color=(255, 255, 255))
denormalize_coordinates = mp_drawing._normalized_to_pixel_coordinates

# TIMER VARIABLES
last_position_head = None
last_time = datetime.now()
distraction_positions = ["ARRIBA", "ABAJO", "IZQ", "DER"]
distraction_threshold = parameters["distraction_seconds"]
status_distraccion = False
distraccion_loop_active = True

last_position_eyes = None
last_time_fatiga = datetime.now()
fatiga_threshold = parameters["fatiga_seconds"]
status_fatiga = False
fatiga_active_loop = True

# Time variables for PERCLOS
start_time = time.time()
closed_time = 0
total_time = 0
status_perclos = False

# FATIGA PERCLOS
PERCLOS_THRESHOLD_TIME = parameters["perclos_time"]
CHECK_INTERVAL = parameters["perclos_interval"]
perclos_percentage = parameters["perclos_percentage"]
MAR_THRESHOLD = parameters["mar_value"]
yawn_count = 0

def enviar_spi(spi,string_to_send):
    #l = [hex(byte) for byte in string_to_send]
    #l = [int(byte, 16) for byte in l]
    l= list(string_to_send)
    resp = spi.writebytes(l)

# ser = serial.Serial(
#     port='/dev/ttyAMA4',  # or '/dev/ttyS0' depending on your configuration
#     baudrate=9600,        # Set baud rate
#     parity=serial.PARITY_NONE,
#     stopbits=serial.STOPBITS_ONE,
#     bytesize=serial.EIGHTBITS,
#     timeout=1             # Read timeout in seconds)
# )


#spi = SpiDev()
#spi.open(1,2) #SPI1 - CS2
#spi.max_speed_hz = 4000
# Crea una ventana de OpenCV
cv2.namedWindow(window_name, cv2.WINDOW_NORMAL)
cv2.setWindowProperty(window_name, cv2.WND_PROP_FULLSCREEN, 1)
contador_envios = 0
cinturon_anterior = True
lentes_anterior = True
distraccion_anterior = False
cellphone_anterior = False
cigarrete_anterior = False

while True:
    try:     
        if contador_envios >25:
            
            if  status_fatiga == True or \
                (status_glasses == False and lentes_anterior == True) or \
                (status_cinturon == False and cinturon_anterior == True) or \
                (status_cellphone == True and cellphone_anterior == False) or \
                (status_cigarrete == True and cigarrete_anterior == False) or \
                (status_distraccion == True and distraccion_anterior == False):
                
                envio = f"DSM,C:{int(status_cellphone)},G:{int(status_cigarrete)},L:{int(status_glasses)},D:{int(status_distraccion)},F:{int(status_fatiga)},T:{int(status_cinturon)},O:{current_status_eyes[:2]},M:{status_head[:2]}\r\n".encode()
                #ser.write(envio)
                #ser.flush()
                #*enviar_spi(spi,envio)

                #*print("Enviado:",envio)
                #*print("Tamaño",len(envio))

                #*response = ser.read(1)
                # if response:
                #     print(f"Recibido: '{response.decode()}'")
                # else:
                #     print("Sin respuesta"
                
            cellphone_anterior = status_cellphone
            cigarrete_anterior = status_cigarrete
            lentes_anterior = status_glasses
            cinturon_anterior = status_cinturon
            distraccion_anterior = status_distraccion
            
            contador_envios = 0
        if len(data_validada)>0 and capturing == False:
                envio_QBAR = f"{data_validada}\r\n".encode()
                data_validada = ""
                #time.sleep(0.02)
                #ser.write(envio_QBAR)
                #ser.flush()
        contador_envios+=1
    #except serial.SerialException as e:
    except Exception as e:
        print(f"Error: {e}")
        #ser.close()
        #spi.close()


    ret, im = cam.read()
    im = cv2.resize(im, (WIDTH, HEIGHT))
    im = cv2.flip(im, 1)
    imRGB = cv2.cvtColor(im, cv2.COLOR_BGR2RGB)
    imTensor = vision.TensorImage.create_from_array(imRGB)
    detections = detector.detect(imTensor)
    image = utils.visualize(im, detections)

    results = mp_facemesh.process(im)
    faces_found = results.multi_face_landmarks
    
    face_2d = []
    face_3d = []

    current_time = datetime.now()
    current_time_fatiga = current_time
    

    if faces_found != None:

        # Face ID
        if face_id_contador == 0:
            small_frame = cv2.resize(im, (0, 0), fx=0.25, fy=0.25)
            code = cv2.COLOR_BGR2RGB
            rgb_small_frame = cv2.cvtColor(small_frame, code)
            
            # Find all people in the image using a trained classifier model
            # Note: You can pass in either a classifier file name or a classifier model instance
            predictions = predict(rgb_small_frame, model_path=FACE_ID_MODEL)

            # Print results on the console
            for name, (top, right, bottom, left) in predictions:
                persona_id = name

            face_id_contador = 100

        # process_this_frame = not process_this_frame
        face_id_contador = face_id_contador - 1

        for this_face_landmark in faces_found:

            # GLASSES
            try:
                status_glasses = get_glasses(im, this_face_landmark.landmark, WIDTH, HEIGHT)
            except:
                status_glasses = False

            # HEAD POSITION & DISTRACTION 
            face_2d, face_3d, p1, p2, x, y, z, status_head = get_head(this_face_landmark.landmark, face_2d, face_3d, HEIGHT, HEIGHT)

            # DEBUG - DISTRACCION -------
            if status_head in distraction_positions:
                if distraccion_loop_active == False:
                    last_position_head = status_head
                    last_time = current_time
                    distraccion_loop_active = True
                
                else: 
                    if (current_time - last_time).seconds > distraction_threshold:
                        #last_position_head = status_head
                        #last_time = current_time
                        if last_position_head in distraction_positions:
                            distraccion_loop_active = False
                            status_distraccion = True
                        else:
                            status_distraccion = False
                            distraccion_loop_active = True
                            
            else:
                status_distraccion = False
                distraccion_loop_active = False

            # DEBUG - DISTRACCION FINA -------

            cv2.line(im, p1, p2, (0, 0, 0), 1)

            # EAR & DROWNSINESS
            status_ear, _ = calculate_avg_ear(
                this_face_landmark.landmark,
                chosen_left_eye_idxs,
                chosen_right_eye_idxs,
                WIDTH,
                HEIGHT,
            )

            mouth = np.array([(int(this_face_landmark.landmark[i].x * WIDTH), int(this_face_landmark.landmark[i].y * HEIGHT)) for i in MOUTH_INDEXES])
            status_mar = calculate_MAR(mouth)
            
            # Check if mouth is open (yawning detection based on MAR)
            if status_mar > MAR_THRESHOLD:
                yawn_count += 1

            if status_ear <= parameters["ear_value"]:
                current_status_eyes = "CERRADOS"
            else:
                current_status_eyes = "ABIERTOS"
            
            # FATIGA NORMAL
            if current_status_eyes == "CERRADOS":
                if fatiga_active_loop == False:
                    last_position_eyes = current_status_eyes
                    last_time_fatiga = current_time_fatiga
                    fatiga_active_loop = True
                else: 
                    if (current_time_fatiga - last_time_fatiga).seconds > fatiga_threshold:
                        if last_position_eyes == "CERRADOS":
                            if last_position_eyes == current_status_eyes:
                                status_fatiga = True
                                fatiga_active_loop = False
                            else:
                                status_fatiga = False
            else:
                fatiga_active_loop = False
                status_fatiga = False
            

            # FATIGA PERCLOS
            if current_status_eyes == "CERRADOS":
                closed_time += CHECK_INTERVAL
            
            total_time += CHECK_INTERVAL

            if total_time >= PERCLOS_THRESHOLD_TIME:
                perclos = (closed_time / total_time) * 100
                print(f"{datetime.now()} -- PERCLOS over {PERCLOS_THRESHOLD_TIME} seconds: {perclos:.2f}%")

                if perclos > perclos_percentage or yawn_count >= parameters["num_bostesos"]:
                    status_perclos = True
                else:
                    status_perclos = False
                yawn_count = 0
                closed_time = 0
                total_time = 0
                start_time = time.time()
      
        # OBJECT DETECTIONS
        if len(detections.detections) > 0:
            for i in detections.detections:
                if i.categories[0].category_name == 'celular':
                    status_cellphone = True
                elif i.categories[0].category_name == 'cigarro':
                    status_cigarrete = True
                elif i.categories[0].category_name == 'cinturon':
                    status_cinturon = True
        else:
            status_cellphone = False
            status_cigarrete = False
            status_cinturon = False


        # DRAW FACE LANDMARKS
        mp_drawing.draw_landmarks(
        im,
        this_face_landmark,
        mp.solutions.face_mesh.FACEMESH_CONTOURS,
        mp_circle,
        mp_line,
        )

    utils.display(im, str(int(fps))+' FPS', 1)
    utils.display(im, f"CELULAR: {status_cellphone}", 2)
    utils.display(im, f"LENTES: {str(status_glasses)}", 3)
    utils.display(im, f"CINTURON:{status_cinturon}", 4)
    utils.display(im, f"CIGARRO: {status_cigarrete}", 5)

    #utils.display(im, "X: " + str(np.round(x, 2)), 4)
    #utils.display(im, "Y: " + str(np.round(y, 2)), 5)
    #utils.display(im, "Z: " + str(np.round(z, 2)), 6)
    utils.display(im, f"OJOS: {current_status_eyes}", 6)
    utils.display(im, f"EAR: {round(status_ear, 2)}", 7)
    utils.display(im, f"MAR: {round(status_mar, 2)}", 8)
    utils.display(im, f"MIRADA: {status_head}", 9)
    utils.display(im, f"DISTRACCION: {status_distraccion}", 10)
    utils.display(im, f"Ultimo PERCLOS: {status_perclos}", 11)
    utils.display(im, f"FATIGA: {status_fatiga}", 12)
    utils.display(im, f"QR/BAR:{data_validada}", 13)
    utils.display(im, f"CONDUCTOR:{persona_id}", 14)
    

    cv2.imshow(window_name, im)

    # if cv2.waitKey(1) == ord('q'):
    #     break
    if cv2.waitKey(1) == 27:  # 27 es el código ASCII de la tecla Escape
        break
    tEnd = time.time()
    loopTime = tEnd-tStart
    fps = .9*fps + .1*1/loopTime
    tStart = time.time()
cv2.destroyAllWindows()
#spi.close()