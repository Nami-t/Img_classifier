import cv2 as cv
import numpy as np
import matplotlib.pyplot as plt
from tensorflow.keras import datasets, layers, models

# Load and normalize the CIFAR-10 dataset
(training_images, training_labels), (testing_images, testing_labels) = datasets.cifar10.load_data()
training_images, testing_images = training_images / 255.0, testing_images / 255.0

# Class names for CIFAR-10 dataset
class_names = ['Plane', 'Car', 'Bird', 'Cat', 'Deer', 'Dog', 'Frog', 'Horse', 'Ship', 'Truck']

# # Plot the first 16 images from the training set
# for i in range(16):
#     plt.subplot(4, 4, i + 1)
#     plt.xticks([])  # Remove x-axis ticks
#     plt.yticks([])  # Remove y-axis ticks
#     plt.imshow(training_images[i], cmap=plt.cm.binary)  # Display the image
#     plt.xlabel(class_names[training_labels[i][0]])  # Label the image

# plt.show()

# Slice the dataset if needed
training_images = training_images[:20000]
training_labels = training_labels[:20000]
testing_images = testing_images[:4000]
testing_labels = testing_labels[:4000]

model = models.load_model('image_classifier.keras')
# Load and preprocess the input image
img = cv.imread('./deer-1.jpg')
img = cv.cvtColor(img, cv.COLOR_BGR2RGB)

# Resize the image to 32x32 as CIFAR-10 images are of size 32x32
img_resized = cv.resize(img, (32, 32))

# Display the image
plt.imshow(img_resized, cmap=plt.cm.binary)
plt.show()

# Make a prediction
img_array = np.array([img_resized]) / 255.0  # Normalize the image
prediction = model.predict(img_array)
index = np.argmax(prediction)

# Print the predicted class
print(f'Prediction: {class_names[index]}')
