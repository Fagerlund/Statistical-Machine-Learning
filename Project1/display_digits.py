# Display images from the digits data set

import matplotlib.pyplot as plt
from sklearn import datasets

# import some data to play with
digits = datasets.load_digits()

im = digits.images[1]
print(digits.images.shape)
plt.gray() 
plt.matshow(digits.images[1])
plt.matshow(digits.images[10])
plt.show() 
