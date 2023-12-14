from keras import backend
import numpy as np
import cv2
import os

def l1_distance(self, inputs):
    return backend.abs(inputs[0] - inputs[1])

def l1_distance_output_shape(self, shapes):
    assert shapes[0] == shapes[2]
    return (1,)

def cos_sim(self, model, feature_vec, labels, filename):

    image       = np.array(cv2.imread(os.path.join(os.path.abspath('./datasets/mars/'), filename)))
    input_fvec  = model.predict(np.expand_dims(image, axis=0))
    similarity  = np.sum(feature_vec * input_fvec, axis=1)
    similarity  = similarity / np.linalg.norm(feature_vec, axis=1)
    similarity  = similarity / np.linalg.norm(input_fvec)

    return [np.argmax(labels[i]) for i in np.argsort(similarity)[:6]]