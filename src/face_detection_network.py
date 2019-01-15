from keras.models import Model
from keras.optimizers import Adam
from keras.models import load_model
from keras.layers import Input, Dense, Flatten, Conv2D, MaxPooling2D, BatchNormalization
from keras.callbacks import EarlyStopping, TensorBoard, ModelCheckpoint
import os


class FaceNetwork:
    def __init__(self, lr, loss, metrics):
        self.lr = lr
        self.loss = loss
        self.metrics = metrics
        self.input_shape = (227, 227, 3)
        self.model = None

    def create_rcnn_face_detection_network(self):
        inputs = Input(shape=self.input_shape, name='input_tensor')
        # First Convolution
        conv1 = Conv2D(filters=96, kernel_size=(11, 11), strides=(4, 4), padding='valid', activation='relu',
                       name='conv1')(inputs)
        pool1 = MaxPooling2D(pool_size=(3, 3), strides=(2, 2), name='pool1')(conv1)
        pool1 = BatchNormalization()(pool1)

        # Second Convolution
        conv2 = Conv2D(filters=256, kernel_size=(5, 5), strides=(1, 1), padding='same', activation='relu',
                       name='conv2')(pool1)
        pool2 = MaxPooling2D(pool_size=(3, 3), strides=(2, 2), name='pool2')(conv2)
        pool2 = BatchNormalization()(pool2)

        # Third Convolution
        conv3 = Conv2D(filters=384, kernel_size=(3, 3), strides=(1, 1), padding='same', activation='relu',
                       name='conv3')(pool2)
        conv3 = BatchNormalization()(conv3)

        # Fourth Convolution
        conv4 = Conv2D(filters=384, kernel_size=(3, 3), strides=(1, 1), padding='same', activation='relu',
                       name='conv4')(conv3)
        conv4 = BatchNormalization()(conv4)

        # Fifth Convolution
        conv5 = Conv2D(filters=256, kernel_size=(3, 3), strides=(1, 1), padding='same', activation='relu',
                       name='conv5')(conv4)
        pool5 = MaxPooling2D(pool_size=(3, 3), strides=(2, 2), name='pool5')(conv5)
        pool5 = BatchNormalization()(pool5)

        # Flatten the output of fifth covolution part
        flatten = Flatten(name='flatten')(pool5)

        # Fully connected with 4096 units
        fully_connected = Dense(4096, activation='relu', name='fully_connected')(flatten)

        # Fully connected with 512 units
        face_detection = Dense(512, activation='relu', name='detection')(fully_connected)

        # Face detection output with 2 units
        face_output = Dense(2, name='face_detection_output')(face_detection)

        self.model = Model(inputs=inputs, outputs=face_output)

        self.model.compile(Adam(lr=self.lr), loss=self.loss, metrics=self.metrics)
        return self.model

    def define_callbacks(self):
        path_checkpoint = 'face_rnn_checkpoint.keras'
        callback_checkpoint = ModelCheckpoint(filepath=path_checkpoint, monitor='val_loss', verbose=1,
                                              save_weights_only=True, save_best_only=True)

        callback_early_stopping = EarlyStopping(monitor='val_loss', patience=3, verbose=1)
        log_dir = 'face_rnn_logs'
        callback_tensorboard = TensorBoard(log_dir=log_dir, histogram_freq=0, write_graph=True)
        return [callback_checkpoint, callback_early_stopping, callback_tensorboard]

    def train_face_detection_model(self, X_train, y_Face, save_path):
        print('\n\nStarted Face detection model training')
        callbacks = self.define_callbacks()
        self.model.fit(x=X_train, y=y_Face, batch_size=32, epochs=3, callbacks=callbacks)
        self.model.save(save_path)
        print('\n\nModel training is done and save to {0} file'.format(save_path))

    def load_face_model(self, file_name):
        exists = os.path.isfile(file_name)
        if exists:
            self.model = load_model(filepath=file_name)
