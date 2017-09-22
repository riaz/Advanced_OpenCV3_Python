import sys
import os
import dlib
import glob
from skimage import io


PREDICTOR_PATH = os.path.join("..", "..", "dependencies", "shape_predictor_68_face_landmarks.dat")
FACE_RECOG_MODEL_PATH = os.path.join("..","..","dependencies","dlib_face_recognition_resnet_model_v1 2.dat")
FACE_IMAGES = os.path.join("..","images","faces","*.jpg")


# Loading all the models and the detectors to find the faces in the images.
# we will be using the shape predictor to find the landmarks in a image, which has various applications

detector = dlib.get_frontal_face_detector()
shape_pred = dlib.shape_predictor(PREDICTOR_PATH)
face_rec = dlib.face_recognition_model_v1(FACE_RECOG_MODEL_PATH)


win = dlib.image_window()


# Now process all the images
if __name__ == '__main__':
    for f in glob.glob(FACE_IMAGES):
        print(" Processing file : {}".format(f))

        img = io.imread(f)

        win.clear_overlay()
        win.set_image(img)

        #ask the detector to find the bounding boxes of each face
        # The 1 in the argument indicates that we should unsample the image 1 time.
        # This will make everything bigger and allow us to detect more faces

        dets = detector(img, 1)

        print("Number of faces detected : {}".format(len(dets)))

        # We will now process each image that we have found
        for k,d in enumerate(dets):
            print("Detection {} : Left: {} Top: {} Right: {} Bottom: {}".format(
                k, d.left(), d.top(), d.right(), d.bottom()))

            # getting the landmark points in the face
            shape = shape_pred(img, d)

            # drawing the landmark points in the face
            win.clear_overlay()
            win.add_overlay(d)  # this adds the box
            win.add_overlay(shape) # this adds the landmark points


            # we will now try to recognize the face in the image
            face_detector = face_rec.compute_face_descriptor(img, shape)
            print(face_detector) # this is a 128D vector of the face


            dlib.hit_enter_to_continue()
