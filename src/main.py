#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sat Jan  5 11:53:36 2019

@author: ShivamMac
"""

from face_detection_network import FaceNetwork
from hyperface_network import HyperfaceNetwork
from region_proposals import RegionProposals
import numpy as np
import matplotlib.pyplot as plt
from keras.utils import to_categorical
from keras import backend as K
import scipy
import os

os.chdir("..")

train_face_detection_network = True


class Hyperface:
    def get_data(self, path):
        self.data = np.load(path)
        return self.data

    def plot_image_data(self, num_image=18):
        images = self.data[:num_image, 0]
        face_label = self.data[:num_image, 1]
        gender_label = self.data[:num_image, 5]

        fig, axes = plt.subplots(3, 6, figsize=(20, 10))
        fig.subplots_adjust(hspace=0.3, wspace=0.3)
        for i, ax in enumerate(axes.flat):
            ax.imshow(images[i])
            face = 'Face' if face_label[i][0] else 'No Face'
            gender = 'Male' if np.array_equal(gender_label[i], [1, 0]) else 'Female' if np.array_equal(gender_label[i],[0, 1]) else 'No Face'
            shape = 'Shape: ' + str(images[i].shape)
            xlabel = face + '\n' + gender + '\n' + shape
            ax.set_xlabel(xlabel)
            ax.set_xticks([])
            ax.set_yticks([])
        plt.show()

    # Creating and resizing training data
    def resize_images_to_227_x_227(self, images):
        images = images / 255
        return np.array([scipy.misc.imresize(im, (227, 227)) for im in images])

    def get_data_and_print_shapes(self):
        X_train = self.resize_images_to_227_x_227(self.data[:, 0])
        print("training data shape: ", X_train.shape)

        # You can change the array in the second element from [1, 1] to [1] and [0, 0] to [0] .
        y_Face = np.array([face[0] for face in self.data[:, 1]])
        print("Face label shape: ", y_Face.shape)

        y_Landmarks = np.array([mark for mark in self.data[:, 2]])
        print("Landmarks label shape: ", y_Landmarks.shape)

        y_Visibility = np.array([visible for visible in self.data[:, 3]])
        print("Visibility label shape: ", y_Visibility.shape)

        y_Pose = np.array([pose for pose in self.data[:, 4]])
        print("Pose label shape: ", y_Pose.shape)

        y_Gender = np.array([1 if np.array_equal(gender, [1, 0]) else 0 if np.array_equal(gender, [0, 1]) else 2 for gender in self.data[:, 5]])
        y_Gender = to_categorical(y_Gender)[:, :-1]
        print("Gender label shape: ", y_Gender.shape)
        return X_train, y_Face, y_Landmarks, y_Visibility, y_Pose, y_Gender

    def visibility_factor_for_localization_loss(self, visibility):
        expand_visibility = np.tile(np.expand_dims(visibility, axis=2), [1, 1, 2])
        return np.reshape(expand_visibility, [expand_visibility.shape[0], -1])

    def custom_localization_loss(self, visibility):
        visibility = self.visibility_factor_for_localization_loss(visibility)
        visibility = K.variable(visibility)

        def wrapper_loss(y_true, y_pred):
            return K.mean(visibility * K.square(y_pred - y_true), axis=-1)

        return wrapper_loss


if __name__ == '__main__':
    hyperface = Hyperface()
    data = hyperface.get_data('data/data.npy')
    # hyperface.plot_image_data(num_image=18)

    X_train, y_Face, y_Landmarks, y_Visibility, y_Pose, y_Gender = hyperface.get_data_and_print_shapes()

    face_network = FaceNetwork(lr=0.0001, loss='sparse_categorical_crossentropy', metrics=['accuracy'])
    region_of_proposals = RegionProposals()
    selected_regions = region_of_proposals.get_region_of_proposals(data[:, 0][1])
    print('\n' + str(selected_regions.shape) + ' regions are selected with IOU greater than 0.5')

    # Training here is done using actual image, but it should be trained on region images prepared in above step.
    if train_face_detection_network:
        face_network.create_rcnn_face_detection_network()
        face_network.train_face_detection_model(X_train=X_train, y_Face=y_Face, save_path='face_model.h5')
    else:
        face_network.load_face_model(file_name='face_model.h5')
    print('\nface detection network\n')
    face_network.model.summary()

    localization_loss = hyperface.custom_localization_loss(y_Visibility)

    hyperface_loss = {'face_detection_out': 'sparse_categorical_crossentropy',
                      'landmarks_output': localization_loss, 'visibility_output': 'mean_squared_error',
                      'pose_output': 'mean_squared_error', 'gender_output': 'categorical_crossentropy'}
    hyperface_loss_weights = {'face_detection_out': 1, 'landmarks_output': 5,
                              'visibility_output': 0.5, 'pose_output': 5, 'gender_output': 2}

    hyperface_model = HyperfaceNetwork(lr=0.0001, loss=hyperface_loss, loss_weights=hyperface_loss_weights,
                                       metrics=['accuracy'])
    hyperface_model.create_hyperface_network()
    hyperface_model.initialize_weights_of_hyperface_with_face_detection_layer_weights(face_network.model)
    print('\nHyperface model network\n')
    hyperface_model.model.summary()
    hyperface_model.train_hyperface_model(X_train=X_train, labels=[y_Face, y_Landmarks, y_Visibility, y_Pose, y_Gender],
                                          save_path='hyperface_model.h5')
