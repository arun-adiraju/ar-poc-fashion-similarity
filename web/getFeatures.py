import os
import re

import tensorflow as tf
from tensorflow.python.platform import gfile

import numpy as np


print('TensorFlow Verions: %s' % tf.__version__)

model_dir = '../tf_files/'

def create_graph():
  with gfile.FastGFile(os.path.join(model_dir, 'retrained_graph.pb'), 'rb') as f:
    graph_def = tf.GraphDef()
    graph_def.ParseFromString(f.read())
    _ = tf.import_graph_def(graph_def, name='')


def extract_features(file_name):
  nb_features = 2048
  features = []#np.empty((len(list_images),nb_features)) # Deep Features to be stored here
  labels = [] # Image class labels to be stored here

  create_graph() 

  with tf.Session() as sess:
    next_to_last_tensor = sess.graph.get_tensor_by_name('pool_3:0') # We want the output features from this layer of Inception model
    
    image_data = gfile.FastGFile(file_name, 'rb').read()
    predictions = sess.run(next_to_last_tensor,
                             {'DecodeJpeg/contents:0': image_data})
    features = np.squeeze(predictions) # Store the output deep features for images
    #labels.append(re.split('_\d+',image.split('/')[1])[0]) # Get class label based on file names: Class + '_' + Digits + .jpg|JPG
    print(features)

    return features