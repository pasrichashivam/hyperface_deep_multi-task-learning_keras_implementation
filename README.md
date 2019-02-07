## This repository is the implementation of research paper for Multi-task Learning using Keras.
* **Title**: HyperFace: A Deep Multi-task Learning Framework for Face Detection, Landmark Localization, Pose Estimation, and Gender Recognition
* **Link**: http://arxiv.org/abs/1603.01249v2

## Brief of paper
* We need to perform the following tasks:
  * Face localization
  * Face landmark localization
  * Face landmark visibility estimation
  * Pose prediction (roll, pitch and yaw estimation)
  * Gender prediction


### Points for implementation

* Here 2 model architectures are used.
1. R-CNN for face landmark localization (face-model).
      * Use Selectivesearch algorithm to select ROI (region of proposals).
      * Bounding box is rescaled to 227x227 which will be the input shape of the model.
      * These region of proposals are the inputs to the R-CNN (Face model) to detect the face in images (Face localization).
      * In R-CNN base model used is AlexNet architecture to train.

### Face Model Architecture
   ![face_model](images/face_model.png?raw=true "face_model")

### Keras Model Architecture
   ![keras_face_model](model_graphs/face_model.png?raw=true "keras_face_model")

   2. Hyperface model for multi-task learning.
      * Basic model Hyperface is the architecture of AlexNet.
      * Except convolutional layers, remove all fully-connected (Dense) layers.
      * Input each bounding box through pre-trained face-model.
      * Initialize the weights of hyperface layers with common face-model layer weights.
  * *Classification and regression*
    * following 5 networks are there:
      1. Face Detection: (2 outputs Yes / No)
      2. Landmark Localization: What are the coordinates of landmark features. (21 landmarks, each 2 values for x,y 42 outputs total)
      3. Landmark Visibility: Which landmarks are visible. (21 yes/no outputs)
      4. Pose Prediction: Roll, pitch, yaw of the face. (3 outputs)
      5. Gender Prediction: Male or female (2 outputs).
    * Each network is connected to FC- layer with 512 units, followed by anothe Dense layer having units correponding to each task above mentioned.

### Hyperface architecture
   ![hyperface_model](images/hyperface_model.png?raw=true "hyperface_model")

### Keras Model Architecture
   ![keras_hyperface_model](model_graphs/hyperface_model.png?raw=true "keras_hyperface_model")

### Cost functions for different tasks.
  1. Face Detection: Cross Entropy
  2. Landmark Localization: Custom loss (Landmark Visibility * Mean squared Error).
  3. Landmark Visibility: Mean squared Error
  4. Pose Prediction: Mean squared Error
  5. Gender Prediction: Cross Entropy.

## Running the source code.
* install the packages in requirements file using command.
  > pip install -r requirements.txt
* Move inside src folder and run it using command.
  > python src/main.py



  
