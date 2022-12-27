import pandas as pd
import numpy as np
import tensorflow as tf
import tensorflow_hub as hub
import os
from keras.applications import imagenet_utils
from keras.utils.image_utils import img_to_array
from PIL import Image

labels_csv = pd.read_csv("labels.csv")
labels = labels_csv["breed"].to_numpy()
unique_breeds = np.unique(labels)



def load_model(model_path):
  """
  Loads a saved model from a specified path.
  """
  print(f"Loading saved model from: {model_path}")
  model = tf.keras.models.load_model(model_path,
                                     custom_objects={"KerasLayer":hub.KerasLayer})
  return model


loaded_full_model = load_model("1-all-images-Adam.h5")


def process_image(image_path):
  """
  Takes an image file path and turns it into a Tensor.
  """
  
  # image = tf.io.read_file(image_path)
  
  # image = tf.image.decode_jpeg(image_path, channels=3)
  
  image = tf.image.convert_image_dtype(image_path, tf.float32)
  print(image.shape)
  image = tf.image.resize(image, size=[480, 480])
  return image


def create_data_batches(x, y=None, batch_size=32):
  """
  Creates batches of data out of image (x) and label (y) pairs.
  Shuffles the data if it's training data but doesn't shuffle it if it's validation data.
  Also accepts test data as input (no labels).
  """
  
  print("Creating data batches...")
  data = tf.data.Dataset.from_tensor_slices((tf.constant(x)))
  data_batch = data.map(process_image).batch(batch_size)
  return data_batch

def get_pred_label(prediction_probabilities):
  """
  Turns an array of prediction probabilities into a label.
  """
  return unique_breeds[np.argmax(prediction_probabilities)]

def predict():
  custom_image_paths = tf.keras.preprocessing.image.load_img('D:/k-digital/source/web_mk2/template1/uploads/upload.jpg')
  custom_data = create_data_batches(custom_image_paths)
  custom_preds = loaded_full_model.predict(custom_data)
  custom_pred_labels = [get_pred_label(custom_preds)]
  pic = custom_data.unbatch().as_numpy_iterator()
  return pic, custom_pred_labels

def prepare_image(image, target):
  # if the image mode is not RGB, convert it
  if image.mode != "RGB":
    image = image.convert("RGB")

  # resize the input image and preprocess it
  image = image.resize(target)
  image = img_to_array(image)
  image = np.expand_dims(image, axis=0)
  image = imagenet_utils.preprocess_input(image)

  # return the processed image
  return image