from keras.layers import Input, Conv2D, Lambda, Dense, Flatten, MaxPooling2D
from keras.models import Model, Sequential
from keras.regularizers import l2
from keras import backend as K
from keras.optimizers import Adam
import numpy.random as rng


class SiameseNet:
    """
    Set up siamese network for training. Able to train from scratch or train from saved weights
    Use method 'setup' to return the network.
    """

    def __init__(self, weight_path=None):

        if weight_path is None:
            def W_init(shape, name=None):
                """Initialize weights as in paper"""
                values = rng.normal(loc=0, scale=1e-2, size=shape)
                return K.variable(values, name=name)

            def b_init(shape, name=None):
                """Initialize bias as in paper"""
                values = rng.normal(loc=0.5, scale=1e-2, size=shape)
                return K.variable(values, name=name)
        else:  # Default settings for keras.layers.Conv2D and keras.layers.Dense
            W_init = 'glorot_uniform'
            b_init = 'zeros'

        input_shape = (105, 105, 1)

        # build convnet to use in each siamese 'leg'
        convnet = Sequential()
        convnet.add(Conv2D(64, (10, 10), activation='relu', input_shape=input_shape,
                           kernel_initializer=W_init, kernel_regularizer=l2(2e-4)))
        convnet.add(MaxPooling2D())
        convnet.add(Conv2D(128, (7, 7), activation='relu',
                           kernel_regularizer=l2(2e-4), kernel_initializer=W_init, bias_initializer=b_init))
        convnet.add(MaxPooling2D())
        convnet.add(Conv2D(128, (4, 4), activation='relu',
                           kernel_regularizer=l2(2e-4), kernel_initializer=W_init, bias_initializer=b_init))
        convnet.add(MaxPooling2D())
        convnet.add(Conv2D(256, (4, 4), activation='relu',
                           kernel_regularizer=l2(2e-4), kernel_initializer=W_init, bias_initializer=b_init))
        convnet.add(Flatten())
        convnet.add(Dense(4096, activation="sigmoid",
                          kernel_regularizer=l2(1e-3), kernel_initializer=W_init, bias_initializer=b_init))

        left_input = Input(input_shape)
        right_input = Input(input_shape)
        # call the convnet Sequential model on each of the input tensors so params will be shared
        encoded_l = convnet(left_input)
        encoded_r = convnet(right_input)
        # layer to merge two encoded inputs with the l1 distance between them
        L1_layer = Lambda(lambda tensors: K.abs(tensors[0] - tensors[1]))
        # call this layer on list of two input tensors.
        L1_distance = L1_layer([encoded_l, encoded_r])
        prediction = Dense(1, activation='sigmoid', bias_initializer=b_init)(L1_distance)
        self._siamese_net = Model(inputs=[left_input, right_input], outputs=prediction)

        self._siamese_net.compile(loss="binary_crossentropy", optimizer=Adam(0.00006))

        # Load weights from weights file if we assign one
        if weight_path is not None:
            self._siamese_net.load_weights('weights/siamese_weights_98000.h5')

    def setup(self):
        """:return well-established network object"""
        return self._siamese_net
