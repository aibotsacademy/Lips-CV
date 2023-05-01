import dlib
import cv2
import math

# Define the path to the shape predictor
shape_predictor_path = '/shape_predictor_68_face_landmarks.dat'

# Initialize the face detector, shape predictor, and the image
detector = dlib.get_frontal_face_detector()
predictor = dlib.shape_predictor(shape_predictor_path)
img = cv2.imread('/h4A.jpg')

# Convert the image to grayscale
gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

# Detect faces in the grayscale image
faces = detector(gray, 1)

# Loop over the detected faces
for face in faces:
    # Get the facial landmarks
    landmarks = predictor(gray, face)

    # Get the x, y coordinates of the upper lip
    upper_lip_left_x = landmarks.part(52).x
    upper_lip_left_y = landmarks.part(52).y
    upper_lip_right_x = landmarks.part(67).x
    upper_lip_right_y = landmarks.part(67).y

    # Calculate the physical distance between the left and right points of upper lip in millimeters, assuming a scale of 1 pixel = 1 millimeter
    scale = 0.26  # change this to the actual scale of the image
    dist_mm = scale * math.sqrt((upper_lip_right_x - upper_lip_left_x)**2 + (upper_lip_right_y - upper_lip_left_y)**2)

    # Print the calculated distance
    print('Size of the upper lip: {:.2f} mm'.format(dist_mm))
