from keras import backend as K


class LocalizationCustomLoss:
    def __init__(self, batch_size):
        self.batch_size = batch_size
        self.cur_index = 0

    def custom_localization_loss(self, visibility):
        visibility = K.variable(visibility[self.cur_index:self.cur_index + self.batch_size, :])
        self.cur_index += self.batch_size

        def wrapper_loss(y_true, y_pred):
            return K.mean(visibility * K.square(y_pred - y_true), axis=-1)

        return wrapper_loss
