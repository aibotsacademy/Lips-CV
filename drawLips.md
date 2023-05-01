!pip install mediapipe
import cv2
import mediapipe as mp
mp_drawing = mp.solutions.drawing_utils

from google.colab.patches import cv2_imshow

mp_face_mesh = mp.solutions.face_mesh

# Load the image
img = cv2.imread('/h3A.jpg')

# Initialize the Face Mesh model
with mp_face_mesh.FaceMesh() as face_mesh:
  # Process the image and detect the landmarks
  results = face_mesh.process(cv2.cvtColor(img, cv2.COLOR_BGR2RGB))

  # Draw the landmarks on the image
  if results.multi_face_landmarks:
    for face_landmarks in results.multi_face_landmarks:
      mp_drawing.draw_landmarks(img, face_landmarks, mp_face_mesh.FACEMESH_LIPS,
                                mp_drawing.DrawingSpec((255, 255, 255), 1, 1),
                                mp_drawing.DrawingSpec((0, 255, 0), 1, 1))
    
# Display the resulting image
cv2_imshow(img)
