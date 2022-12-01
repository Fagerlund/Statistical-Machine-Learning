import numpy
from sklearn import datasets
import tensorflow as tf
from tensorflow import keras
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder


# Import data
digits = datasets.load_digits()

# Assign input and output
X, y = digits.images, digits.target

# Scale data and make sure it is float32
X = X / 16
X = X.astype('float32')

# Split into train and test datasetsq
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.5)
print(X_train.shape, X_test.shape, y_train.shape, y_test.shape)

# Determine the number of input features
n_features = X_train.shape[1] * X_train.shape[2]
print(n_features)

# Define model
model = tf.keras.Sequential([
    keras.layers.Conv2D(32, (2, 2), activation='relu', input_shape=(8, 8, 1), padding='same'),
    keras.layers.MaxPooling2D((2, 2)),
    keras.layers.Conv2D(64, (2, 2), activation='relu'),
    # keras.layers.MaxPooling2D((2, 2)),
    # keras.layers.Conv2D(64, (3, 3), activation='relu'),
    keras.layers.Flatten(),
    keras.layers.Dense(64, activation = 'relu'),
    # keras.layers.Dropout(0.2),
    keras.layers.Dense(10, activation = 'softmax')
])

# Compile model
loss_fn = tf.keras.losses.SparseCategoricalCrossentropy(from_logits=True)
model.compile(optimizer='adam', loss=loss_fn, metrics=['accuracy'])

# Train model
model.fit(X_train, y_train, epochs=5)

# Evaluate accuracy
test_loss, test_acc = model.evaluate(X_test, y_test, verbose=2)
print('Test accuracy:', test_acc)









# print(y[1])
# plt.gray()
# plt.matshow(X[90])
# plt.show()
