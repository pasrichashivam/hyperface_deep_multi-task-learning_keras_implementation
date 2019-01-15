from keras.models import Model
from keras.optimizers import Adam
from keras.layers import Input, Dense, Concatenate, Flatten, Conv2D, MaxPooling2D, BatchNormalization
from keras.callbacks import EarlyStopping, TensorBoard, ModelCheckpoint


class HyperfaceNetwork:
    def __init__(self, lr, loss, loss_weights, metrics, batch_size, epochs):
        self.lr = lr
        self.loss = loss
        self.loss_weights = loss_weights
        self.epochs = epochs
        self.metrics = metrics
        self.batch_size = batch_size
        self.input_shape = (227, 227, 3)
        self.model = None

    def create_hyperface_network(self):
        input = Input(shape=self.input_shape, name='input')

        # First Convolution
        conv1 = Conv2D(filters=96, kernel_size=(11, 11), strides=(4, 4), padding='valid', activation='relu',
                       name='conv1')(input)
        pool1 = MaxPooling2D(pool_size=(3, 3), strides=(2, 2), name='pool1')(conv1)
        # Batch Normalisation before passing it to the next layer
        pool1 = BatchNormalization()(pool1)

        # Extracting low-level details from poo11 layer to fuse (concatenate) it later
        conv1a = Conv2D(filters=256, kernel_size=(4, 4), strides=(4, 4), activation='relu', name='conv1a')(pool1)
        conv1a = BatchNormalization()(conv1a)

        # Second Convolution
        conv2 = Conv2D(filters=256, kernel_size=(5, 5), strides=(1, 1), padding='same', activation='relu',
                       name='conv2')(pool1)
        pool2 = MaxPooling2D(pool_size=(2, 2), strides=(2, 2), name='pool2')(conv2)
        pool2 = BatchNormalization()(pool2)

        # Third Convolution
        conv3 = Conv2D(filters=384, kernel_size=(3, 3), strides=(1, 1), padding='same', activation='relu',
                       name='conv3')(pool2)
        conv3 = BatchNormalization()(conv3)

        # Extracting mid-level details from conv3 layer to fuse (concatenate) it later with high-level pool5 layer.
        conv3a = Conv2D(filters=256, kernel_size=(2, 2), strides=(2, 2), padding='valid', activation='relu',
                        name='conv3a')(conv3)
        conv3a = BatchNormalization()(conv3a)

        # Fourth Convolution
        conv4 = Conv2D(filters=384, kernel_size=(3, 3), strides=(1, 1), padding='same', activation='relu',
                       name='conv4')(conv3)
        conv4 = BatchNormalization()(conv4)

        # Fifth Convolution
        conv5 = Conv2D(filters=256, kernel_size=(3, 3), strides=(1, 1), padding='same', activation='relu',
                       name='conv5')(conv4)
        pool5 = MaxPooling2D(pool_size=(3, 3), strides=(2, 2), name='pool5')(conv5)
        pool5 = BatchNormalization()(pool5)
        # Fuse (concatenate) the conv1a, conv3a, pool5 layers
        concat = Concatenate(axis=-1, name='concat_layer')([conv1a, conv3a, pool5])
        concat = BatchNormalization()(concat)

        # Add convolution to reduce the size of concatenated layers
        conv_all = Conv2D(filters=192, kernel_size=(1, 1), strides=(1, 1), padding='valid', name='conv_all')(concat)

        # Flatten the output of concatenated layer with reduced filters to 192
        flatten = Flatten(name='flatten_layer')(conv_all)

        # Fully connected with 3072 units
        fc_full = Dense(3072, activation='relu', name='fully')(flatten)

        # Split the network into five separate branches with 512 units from fc_full
        # corresponding to the different tasks.
        detection = Dense(512, activation='relu', name='face_detection')(fc_full)
        landmarks = Dense(512, activation='relu', name='landmarks')(fc_full)
        visibility = Dense(512, activation='relu', name='visibility')(fc_full)
        pose = Dense(512, activation='relu', name='pose')(fc_full)
        gender = Dense(512, activation='relu', name='gender')(fc_full)

        # Face detection output with 2 units
        face_output = Dense(2, name='face_detection_out')(detection)
        # Landmark localization output with 42 units
        landmarks_output = Dense(42, name='landmarks_output')(landmarks)
        # Landmark visibility output with 21 units
        visibility_output = Dense(21, name='visibility_output')(visibility)
        # Pose output with 3 units
        pose_output = Dense(3, name='pose_output')(pose)
        # Gender output with 2 units
        gender_output = Dense(2, name='gender_output')(gender)

        self.model = Model(inputs=input,
                           outputs=[face_output, landmarks_output, visibility_output, pose_output, gender_output])

        self.model.compile(Adam(lr=self.lr), loss=self.loss, loss_weights=self.loss_weights, metrics=self.metrics)

    def initialize_weights_of_hyperface_with_face_detection_layer_weights(self, face_detection_model):
        alexnex_common_layers = {layer.name for layer in self.model.layers}.intersection(
            {layer.name for layer in face_detection_model.layers})
        for layer in alexnex_common_layers:
            self.model.get_layer(layer).set_weights(face_detection_model.get_layer(layer).get_weights())
            print('\nWeights of {0} layer in face detection model is initialized in hyperface model'.format(layer))

    def generator(self, x, y, batch_size=32):
        L = x.shape[0]
        while True:
            batch_start = 0
            while batch_start < L:
                batch_x = x[batch_start:batch_start + batch_size, :, :]
                batch_f = y[0][batch_start:batch_start + batch_size]
                batch_l = y[1][batch_start:batch_start + batch_size, :]
                batch_v = y[2][batch_start:batch_start + batch_size, :]
                batch_p = y[3][batch_start:batch_start + batch_size, :]
                batch_g = y[4][batch_start:batch_start + batch_size, :]
                batch_start += batch_size
                yield batch_x, [batch_f, batch_l, batch_v, batch_p, batch_g]

    def define_callbacks(self):
        path_checkpoint = 'hyperface_checkpoint.keras'
        callback_checkpoint = ModelCheckpoint(filepath=path_checkpoint, monitor='val_loss', verbose=1,
                                              save_weights_only=True, save_best_only=True)

        callback_early_stopping = EarlyStopping(monitor='val_loss', patience=3, verbose=1)
        log_dir = 'hyperface_logs'
        callback_tensorboard = TensorBoard(log_dir=log_dir, histogram_freq=0, write_graph=True)
        return [callback_checkpoint, callback_early_stopping, callback_tensorboard]

    def train_hyperface_model(self, X_train, labels, save_path):
        print('\n\nStarted Hyperface model training')
        callbacks = self.define_callbacks()
        self.model.fit_generator(self.generator(X_train, labels, self.batch_size), steps_per_epoch=self.batch_size,
                                 epochs=self.epochs, callbacks=callbacks)
        self.model.save(save_path)
        print('\n\nModel training is done and save to {0} file'.format(save_path))
