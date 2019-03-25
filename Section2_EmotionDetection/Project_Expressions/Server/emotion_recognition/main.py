import cv2
import sys
import os
from constants import *
from emotion_recognition import EmotionRecognition

cascade_classifier = cv2.CascadeClassifier(CASC_PATH)

def brighten(data,b):
     datab = data * b
     return datab    

def format_image(image):
  if len(image.shape) > 2 and image.shape[2] == 3:
    image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
  else:
    image = cv2.imdecode(image, cv2.CV_LOAD_IMAGE_GRAYSCALE)
  faces = cascade_classifier.detectMultiScale(
      image,
      scaleFactor = 1.3,
      minNeighbors = 5
  )

  # None is we don't found an image
  if not len(faces) > 0:
    return None
  max_area_face = faces[0]
  for face in faces:
    if face[2] * face[3] > max_area_face[2] * max_area_face[3]:
      max_area_face = face

  # Chop image to face
  face = max_area_face
  image = image[face[1]:(face[1] + face[2]), face[0]:(face[0] + face[3])]

  # Resize image to network size
  try:
    image = cv2.resize(image, (SIZE_FACE, SIZE_FACE), interpolation = cv2.INTER_CUBIC) / 255.
  except Exception:
    print("[+] Problem during resize")
    return None
   return image

# Load Model
network = EmotionRecognition()
network.build_network()

font = cv2.FONT_HERSHEY_SIMPLEX

feelings_faces = []
for index, emotion in enumerate(EMOTIONS):
  feelings_faces.append(cv2.imread('./emojis/' + emotion + '.png', -1))


if __name__ == '__main__':

    if len(sys.argv) > 2:
        im = sys.argv[1]
    else:
        print("Please pass a image file as argument")
        exit(1);

    image = cv2.imread(im)

    # Predict result with network
    result = network.predict(format_image(image))


SAVE_DIR = os.path.join("..","output");

# Write results to output folder
if result is not None:
     for index, emotion in enumerate(EMOTIONS):

           # Appends a descriptive text of the detected image
           cv2.putText(image, emotion, (10, index * 20 + 20), cv2.FONT_HERSHEY_PLAIN, 0.5, (0, 255, 0), 1);

           # Append a rectangle area against the detect image
           cv2.rectangle(image, (130, index * 20 + 10), (130 + int(result[0][index] * 100), (index + 1) * 20 + 4), (255, 0, 0), -1)

     #Appends the emotion
     face_image = feelings_faces[result[0].index(max(result[0]))]

     cv2.write(os.path.join(SAVE_DIR,"app_output.jpg"),face_image)

     
