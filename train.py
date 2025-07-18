import os
import numpy as np
import tensorflow as tf
import keras
from tensorflow.keras import layers, models
from tensorflow.keras.preprocessing.image import ImageDataGenerator
import matplotlib.pyplot as plt

# Define the path to your dataset
train_dir = 'mnist-pngs-main/train'
test_dir = 'mnist-pngs-main/test'

# Create ImageDataGenerator instances for train and test data
train_datagen = ImageDataGenerator(rescale=1./255)  # Normalize pixel values to [0, 1]
test_datagen = ImageDataGenerator(rescale=1./255)

# Set up the flow_from_directory for loading images
train_generator = train_datagen.flow_from_directory(
    train_dir,
    target_size=(28, 28),  # Resize the images to 28x28 (MNIST dimensions)
    batch_size=32,
    color_mode='grayscale',  # The images are grayscale (black and white)
    class_mode='categorical',  # Multi-class classification (one-hot encoded labels)
    shuffle=True
)

test_generator = test_datagen.flow_from_directory(
    test_dir,
    target_size=(28, 28),
    batch_size=32,
    color_mode='grayscale',
    class_mode='categorical',
    shuffle=False  # No need to shuffle for evaluation
)

# Build the CNN model
model = models.Sequential()

# Convolutional layer 1
model.add(layers.Conv2D(32, (3, 3), activation='relu', input_shape=(28, 28, 1)))
model.add(layers.MaxPooling2D((2, 2)))

# Convolutional layer 2
model.add(layers.Conv2D(64, (3, 3), activation='relu'))
model.add(layers.MaxPooling2D((2, 2)))

# Fully connected layer
model.add(layers.Flatten())
model.add(layers.Dense(64, activation='relu'))

# Output layer (10 units for 10 classes)
model.add(layers.Dense(10, activation='softmax'))

# Compile the model
model.compile(optimizer='adam',
              loss='categorical_crossentropy',
              metrics=['accuracy'])

# Train the model
history = model.fit(
    train_generator,
    epochs=10,
    validation_data=test_generator
)

# Save the model
model.save('mnist_cnn_model.h5')
print("Model saved as 'mnist_cnn_model.h5'")

# Evaluate the model on the test set
test_loss, test_acc = model.evaluate(test_generator)
print(f"Test accuracy: {test_acc:.4f}")

# Plot training and validation accuracy
plt.plot(history.history['accuracy'], label='accuracy')
plt.plot(history.history['val_accuracy'], label='val_accuracy')
plt.xlabel('Epochs')
plt.ylabel('Accuracy')
plt.ylim([0, 1])
plt.legend(loc='lower right')
plt.title("Accuracy over Epochs")
plt.show()

# Plot training and validation loss
plt.plot(history.history['loss'], label='loss')
plt.plot(history.history['val_loss'], label='val_loss')
plt.xlabel('Epochs')
plt.ylabel('Loss')
plt.legend(loc='upper right')
plt.title("Loss over Epochs")
plt.show()
