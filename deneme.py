# Import necessary libraries
import numpy as np
import pandas as pd
import os
from keras.preprocessing.image import ImageDataGenerator
from keras.models import Sequential
from keras.layers import Conv2D, MaxPooling2D, Flatten, Dense, Dropout
import numpy as np
import matplotlib.pyplot as plt
from keras.utils import load_img, img_to_array


# Define directories for training and validation data
train_dir = "data"
validation_dir = "data"


# Image data augmentation and normalization
train_datagen = ImageDataGenerator(rescale=1./255)
validation_datagen = ImageDataGenerator(rescale=1./255)


# Generate batches of augmented data for training and validation
train_generator = train_datagen.flow_from_directory(
    train_dir,
    target_size=(150, 150),  
    batch_size=32,
    class_mode='categorical')
validation_generator = validation_datagen.flow_from_directory(
    validation_dir,
    target_size=(150, 150),  
    batch_size=32,
    class_mode='categorical')


# Define the CNN model architecture
model = Sequential([
    Conv2D(32, (3, 3), activation='relu', input_shape=(150, 150, 3)),
    MaxPooling2D(2, 2),
    Conv2D(64, (3, 3), activation='relu'),
    MaxPooling2D(2, 2),
    Conv2D(128, (3, 3), activation='relu'),
    MaxPooling2D(2, 2),
    Flatten(),
    Dropout(0.5),
    Dense(512, activation='relu'),
    Dense(3, activation='softmax')
])


# Compile the model
model.compile(optimizer='adam',
              loss='categorical_crossentropy',
              metrics=['accuracy'])


# Train the model
history = model.fit(
    train_generator,
    steps_per_epoch=8,  
    epochs=15,
    verbose=1,
    validation_data=validation_generator,
    validation_steps=8)


# Save the trained model
model.save("final_model.h5")


# Plotting training and validation accuracy and loss over epochs
epochs = [i for i in range(15)]
fig, ax = plt.subplots(1,2)
train_accuracy = history.history["accuracy"]
train_loss = history.history["loss"]
validation_accuracy = history.history["val_accuracy"]
validation_loss = history.history["val_loss"]
fig.set_size_inches(20,10)


# Plot training and validation accuracy
ax[0].plot(epochs, train_accuracy, "go-", label="Training Accuracy")
ax[0].plot(epochs, validation_accuracy, "ro-", label="Validation Accuracy")
ax[0].set_title("Training & Validation Accuracy")
ax[0].legend()
ax[0].set_xlabel("Epochs")
ax[0].set_ylabel("Accuracy")

# Plot training and validation loss
ax[1].plot(epochs, train_loss, "g-o", label="Training Loss")
ax[1].plot(epochs, validation_loss, "r-o", label="Validation Loss")
ax[1].set_title("Testing Accuracy & Loss")
ax[1].legend()
ax[1].set_xlabel("Epochs")
ax[1].set_ylabel("Training & Validation Loss")
plt.show()


# Function to predict emotions from images
def predict_images(img_paths, model):
    plt.figure(figsize=(12, 12))
    for i, img_path in enumerate(img_paths):
        # Load image and convert to array
        img = load_img(img_path, target_size=(150, 150))
        img_array = img_to_array(img)
        img_array = np.expand_dims(img_array, axis=0)  # Add batch dimension
        img_array /= 255.0 # Normalize pixel values

        # Make prediction
        prediction = model.predict(img_array)
        
        # Define emotion labels
        labels = ["Angry", "Happy", "Sad"]
        
        # Get predicted emotion
        predicted_class_index = np.argmax(prediction)
        predicted_emotion = labels[predicted_class_index]

        # Plot image with predicted emotion
        plt.subplot(3, 3, i+1)
        plt.imshow(img)
        plt.title(f"Predicted Emotion: {predicted_emotion}")
        plt.axis('off')
    
    plt.tight_layout()
    plt.show()

# Paths to images for prediction
img_paths = [
    'data/Angry/5cd88fd321000035007f6cd2.jpeg',
    'data/Sad/iStock_000001932580XSmall.jpg',
    'data/Happy/guilherme-stecanella-375176-unsplash.jpg',
    'data/Angry/GettyImages-108328246-1024x683.jpg',
    'data/Angry/anger_ruining_898.jpg',
    'data/Angry/iStock_000010998923XSmall.jpg',
    'data/Sad/depositphotos_11207956-stock-photo-thoughtful-man-in-the-living.jpg',
    'data/Sad/very-sad-man-sitting-alone-on-white-background-depressed-young-man-sitting-businessman-vector.jpg',
    'data/Happy/VJdvLa-download-happy-blackman-png.png'
]
# Call the function to predict emotions from images
predict_images(img_paths, model)
