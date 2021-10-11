import numpy as np
import pandas as pd
import sys
from keras.layers import Layer, Input, Lambda, Reshape, Activation, BatchNormalization, Dropout, Add, TimeDistributed, \
    Multiply, AveragePooling2D, MaxPooling2D
from keras.layers import Layer, Input, Dense, Lambda, Flatten, Reshape, Activation, BatchNormalization, Dropout, Add, TimeDistributed, \
    Multiply, Conv2D, AveragePooling2D, MaxPooling2D
from keras.models import Model, load_model
from tensorflow.keras import Sequential
# from tensorflow.keras.layers import Dense, Flatten, Conv2D
from keras import backend as K
from keras.optimizers import Adadelta, SGD, Adam, RMSprop
# from imblearn.keras import BalancedBatchGenerator
# from imblearn.under_sampling import NearMiss
from keras.callbacks import History, ModelCheckpoint


def generator(from_list_x, from_list_y, batch_size):
    assert len(from_list_x) == len(from_list_y)
    total_size = len(from_list_x)

    while True:  # keras generators should be infinite

        for i in range(0, total_size, batch_size):
            yield np.array(from_list_x[i:i + batch_size]), np.array(from_list_y[i:i + batch_size])


class Round(Layer):

    def __init__(self, **kwargs):
        super(Round, self).__init__(**kwargs)

    def get_output(self, train=False):
        X = self.get_input(train)
        return K.round(X)

    def get_config(self):
        config = {"name": self.__class__.__name__}
        base_config = super(Round, self).get_config()
        return dict(list(base_config.items()) + list(config.items()))


class CNN_Classifier():
    def __init__(self, input_shape, num_classes, filter_size=3, n_filters=10, task='classification',
                 load=False, load_dir=r'D:\Users\NanoProject\Sonar_Nano\weights',
                 weight_file='\saved_CNN_clasifier_noise0.03_learn123_test4_3classes_77_2classes_92.1_try2_class_w0.350.350.3.h5'):
        """
        Parameters:
          input_shape: (tuple) tuple of input shape. (e.g. If input is 6s raw waveform with sampling rate = 16kHz, (96000,) is the input_shape)
          num_classes/output_shape: (tuple)tuple of output shape. (e.g. If we want classify the signal into 100 classes, (100,) is the output_shape)
          kernel_size: (integer) kernel size of convolution operations in residual blocks
          dilation_depth: (integer) type total depth of residual blocks
          n_filters: (integer) # of filters of convolution operations in residual blocks
          task: (string) 'classification' or 'regression'
          regression_range: (list or tuple) target range of regression task
          load: (bool) load previous WaveNetClassifier or not
          load_dir: (string) the directory where the previous model exists
        """
        # save task info
        if task == 'classification':
            self.activation = 'softmax'
            self.scale_ratio = 1
        else:
            print('ERROR: wrong task')
            sys.exit()

        # save input info
        if len(input_shape) == 2:
            self.expand_dims = True
        elif len(input_shape) == 3:
            self.expand_dims = False
        else:
            print('ERROR: wrong input shape')
            sys.exit()
        self.input_shape = input_shape

        # save output info
        if len(num_classes) == 1:
            self.time_distributed = False
        elif len(num_classes) == 2:
            self.time_distributed = True
        else:
            print('ERROR: wrong output shape')
            sys.exit()
        self.output_shape = num_classes

        # save hyperparameters of WaveNet
        self.kernel_size = filter_size
        # self.dilation_depth = dilation_depth
        self.n_filters = n_filters
        self.manual_loss = None

        if load is True:
            self.model = self.construct_model()
            self.model.load_weights(load_dir + weight_file)
            try:
                self.prev_history = pd.read_csv(load_dir + '\CNN_classifier_training_history.csv')
                self.start_idx = len(self.prev_history)
            except:
                self.start_idx = 0
                self.history = None
                self.prev_history = None

        else:
            self.model = self.construct_model()
            self.start_idx = 0
            self.history = None
            self.prev_history = None

    def construct_model(self):
        x = Input(shape=self.input_shape, name='original_input')
        if self.expand_dims == True:
            x_reshaped = Reshape(self.input_shape + (1,), name='reshaped_input')(x)
        else:
            x_reshaped = x

        # FIXME - PROBLEM IS HERE!!!!!!!!!!! IN 118
        out = Conv2D(self.n_filters, (self.kernel_size, self.kernel_size), padding='same',
                     input_shape=self.input_shape)(x_reshaped)
        out = BatchNormalization(axis=-1)(out)
        out = Activation("relu")(out)
        out = MaxPooling2D(pool_size=(2, 2), padding='same')(out)
        out = Conv2D(self.n_filters, (self.kernel_size, self.kernel_size), padding='same')(out)
        out = BatchNormalization(axis=-1)(out)
        out = Activation("relu")(out)
        out = MaxPooling2D(pool_size=(2, 2), padding='same')(out)
        out = Flatten()(out)
        out = Dense(38, activation='relu')(out)
        predictions = Dense(self.output_shape[0], activation='softmax')(out)

        if self.scale_ratio != 1:
            predictions = Lambda(lambda x: x * self.scale_ratio, name='output_reshaped')(predictions)
        model = Model(inputs=x_reshaped, outputs=predictions)
        model.summary()
        return model

    def get_model(self):
        return self.model

    def add_loss(self, loss):
        self.manual_loss = loss

    def fit(self, X_train, y_train, X_val=None, y_val=None, BATCH_SIZE=4, epochs=100, optimizer=Adadelta(), save=True,
            save_dir=r'C:\Users\Administrator\PycharmProjects\SonarNano\weights', class_weights=None):
        # set default losses if not defined
        if self.manual_loss is not None:
            loss = self.manual_loss
            metrics = None
        else:
            if self.activation == 'softmax':
                loss = 'categorical_crossentropy'
                metrics = ['accuracy']
            else:
                loss = 'mean_squared_error'
                metrics = None

        # set callback functions
        if save:
            saved = save_dir + "/saved_CNN_clasifier.h5"
            hist = save_dir + '/CNN_classifier_training_history.csv'

            if X_val is None:
                checkpointer = ModelCheckpoint(filepath=saved, monitor='loss', verbose=1, save_weights_only=True,
                                               save_best_only=True)
            else:
                # checkpointer = ModelCheckpoint(filepath=saved, monitor='val_loss', verbose=1, save_weights_only=True,
                #                                save_best_only=True)
                checkpointer = ModelCheckpoint(filepath=saved, monitor='val_accuracy', verbose=1,
                                               save_weights_only=True,
                                               save_best_only=True, mode='max')
                # checkpointer = ModelCheckpoint(filepath=saved, monitor='val_loss', verbose=1, save_weights_only=True,
                # period=5)
            history = History()
            callbacks = [history, checkpointer]
        else:
            callbacks = None

        # compile the model
        self.model.compile(optimizer, loss, metrics)
        try:
            self.history = self.model.fit_generator(generator(X_train, y_train, batch_size=BATCH_SIZE),
                                                    steps_per_epoch=len(X_train) // BATCH_SIZE,
                                                    validation_steps=len(X_val) // BATCH_SIZE, epochs=epochs,
                                                    validation_data=generator(X_val, y_val, batch_size=BATCH_SIZE),
                                                    callbacks=callbacks, initial_epoch=self.start_idx,
                                                    shuffle=True, class_weight=class_weights)
            # y_train = np.array(y_train)
            # y_val = np.array(y_val)
            # X_train = (x[:,:,0] for x in X_train)
            # X_val = (x[:,:,0] for x in X_val)
            # training_generator = BalancedBatchGenerator(X_train, y_train, sampler = NearMiss(), batch_size = BATCH_SIZE, random_state = 42)
            # self.history = self.model.fit_generator(training_generator,
            #                                         steps_per_epoch=len(X_train) // BATCH_SIZE,
            #                                         validation_steps=len(X_val) // BATCH_SIZE, epochs=epochs,
            #                                         validation_data=generator(X_val, y_val, batch_size=BATCH_SIZE),
            #                                         callbacks=callbacks, initial_epoch=self.start_idx,
            #                                         shuffle=True)
        except:
            if save:
                df = pd.DataFrame.from_dict(self.history)
                df.to_csv(hist, encoding='utf-8', index=False)
            raise
            sys.exit()
        return self.history

    def predict(self, x):
        return self.model.predict(x)

    def global_average_pooling(self, x):
        return K.mean(x, axis=(1, 2))

    def global_average_pooling_shape(self, input_shape):
        return (input_shape[0], input_shape[3])

    def get_attention_model(self):
        prev_model = self.model
        prev_model.layers.pop()
        prev_model.layers.pop()
        prev_model.layers.pop()
        prev_model.layers.pop()
        prev_model.layers.pop()
        prev_model.layers.pop()
        prev_model.summary()
        inp = prev_model.input
        out = prev_model.layers[-1].output
        out = Lambda(self.global_average_pooling,
                     output_shape=self.global_average_pooling_shape)(out)
        out = Dense(3, activation='softmax')(out)
        model = Model(inputs=inp, outputs=out)
        sgd = SGD(lr=0.01, decay=1e-6, momentum=0.5, nesterov=True)
        # adam_opt = Adam(learning_rate=0.0005, beta_1=0.9, beta_2=0.999, amsgrad=True)
        # Compile the model
        # for d in range(len(prev_model.layers)):
        #     model.layers[d].trainable = False
        model.compile(loss='categorical_crossentropy', optimizer=sgd, metrics=['accuracy'])
        model.summary()
        self.model = model
        self.start_idx = 0
        self.history = None
        self.prev_history = None
