# Project Objective:
This project aims to create an emotion recognition model using image processing and deep learning techniques. The model will identify and classify emotions in a given image.

# Dataset:
The dataset used in the project consists of images containing different emotions. The dataset has three main emotion classes: Happy, Sad, and Angry. There are approximately an equal number of images for each class. 
Dataset: https://www.kaggle.com/datasets/sanidhyak/human-face-emotions

# Data Preprocessing:
Data preprocessing steps were performed using the ImageDataGenerator class from the Keras library. Images were resized to 150x150 dimensions and normalized. Additionally, data augmentation techniques were applied to diversify the training data.

# Model Architecture:
The model was built using convolutional neural network (CNN). The model includes three convolutional layers (Conv2D) and three pooling layers (MaxPooling2D). Then, two fully connected (Dense) layers were added, and softmax activation was used in the last layer.
![model summary](https://github.com/ismaillburakk/EmotionRecognitionModel/assets/75124682/e4ea2896-1849-4d2e-b6d8-561c96ce3052)

# Training:
The model was trained on training and validation datasets. Adam optimization was used for training, and categorical cross-entropy loss function was selected. The performance of the model was evaluated using the accuracy metric.

# Results:
During the training process, the model's training and validation accuracy increased, and the loss decreased. The accuracy of the model reached approximately 98%. The performance of the model on the validation set was similar to the training accuracy.
![Graphs](https://github.com/ismaillburakk/EmotionRecognitionModel/assets/75124682/cedcec4c-90e0-428b-b8b3-f5a7b527ca2b)

# Testing and Predictions:
The trained model demonstrated the ability to recognize emotions in test images. The model made successful predictions on the test images, correctly classifying the predicted emotion for each image.
![predicted_images](https://github.com/ismaillburakk/EmotionRecognitionModel/assets/75124682/6b59db99-f6cf-4f2b-95ef-d4678a4a24c9)

# Conclusion and Recommendations:
The project successfully created an emotion recognition model. However, training the model on a larger dataset and providing more data diversity could improve performance. Additionally, subjecting the model to more complex architectures and hyperparameter tuning may further enhance its performance.
