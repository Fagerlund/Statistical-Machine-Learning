import tensorflow as tf
# from tensorflow import keras
# from tensorflow.kersas import layers
import numpy
import pandas
from sklearn import datasets


# import some data to play with
digits = datasets.load_digits()

im = digits.images[1]
print(digits.images.shape)