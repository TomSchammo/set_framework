# Author: Decebal Constantin Mocanu et al.;
# Proof of concept implementation of Sparse Evolutionary Training (SET) of Multi Layer Perceptron (MLP) on CIFAR10 using Keras and a mask over weights.
# This implementation can be used to test SET in varying conditions, using the Keras framework versatility, e.g. various optimizers, activation layers, tensorflow
# Also it can be easily adapted for Convolutional Neural Networks or other models which have dense layers
# However, due the fact that the weights are stored in the standard Keras format (dense matrices), this implementation can not scale properly.
# If you would like to build and SET-MLP with over 100000 neurons, please use the pure Python implementation from the folder "SET-MLP-Sparse-Python-Data-Structures"

# This is a pre-alpha free software and was tested with Python 3.5.2, Keras 2.1.3, Keras_Contrib 0.0.2, Tensorflow 1.5.0, Numpy 1.14;
# The code is distributed in the hope that it may be useful, but WITHOUT ANY WARRANTIES; The use of this software is entirely at the user's own risk;
# For an easy understanding of the code functionality please read the following articles.

# If you use parts of this code please cite the following articles:
#@article{Mocanu2018SET,
#  author =        {Mocanu, Decebal Constantin and Mocanu, Elena and Stone, Peter and Nguyen, Phuong H. and Gibescu, Madeleine and Liotta, Antonio},
#  journal =       {Nature Communications},
#  title =         {Scalable Training of Artificial Neural Networks with Adaptive Sparse Connectivity inspired by Network Science},
#  year =          {2018},
#  doi =           {10.1038/s41467-018-04316-3}
#}

#@Article{Mocanu2016XBM,
#author="Mocanu, Decebal Constantin and Mocanu, Elena and Nguyen, Phuong H. and Gibescu, Madeleine and Liotta, Antonio",
#title="A topological insight into restricted Boltzmann machines",
#journal="Machine Learning",
#year="2016",
#volume="104",
#number="2",
#pages="243--270",
#doi="10.1007/s10994-016-5570-z",
#url="https://doi.org/10.1007/s10994-016-5570-z"
#}

#@phdthesis{Mocanu2017PhDthesis,
#title = "Network computations in artificial intelligence",
#author = "D.C. Mocanu",
#year = "2017",
#isbn = "978-90-386-4305-2",
#publisher = "Eindhoven University of Technology",
#}

from __future__ import division
from __future__ import print_function
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from keras.models import Sequential
from keras.layers import Dense, Dropout, Activation, Flatten
from keras import optimizers
import numpy as np
from tensorflow.keras import backend as K
#Please note that in newer versions of keras_contrib you may encounter some import errors. You can find a fix for it on the Internet, or as an alternative you can try other activations functions.
from strategies.base_strategy import BaseSETStrategy
from srelu import SReLU
from keras.datasets import cifar10
from keras.utils import to_categorical

import tensorflow as tf

AUTOTUNE = tf.data.AUTOTUNE

from keras.constraints import Constraint

from strategies.random_set import RandomSET


def zero_pattern_equal(a, b, tol=0.0):
    """
    Returns True if a and b have zeros at exactly the same indices.
    tol allows treating small values as zero.
    """

    if a is None or b is None:
        return False

    a = np.asarray(a)
    b = np.asarray(b)

    if a.shape != b.shape:
        return False

    zero_a = np.abs(a) <= tol
    zero_b = np.abs(b) <= tol

    return np.array_equal(zero_a, zero_b)


old_weights = None


class MaskWeights(Constraint):

    def __init__(self, mask):
        self.mask = mask
        self.mask_var = K.variable(self.mask, K.floatx())

    def __call__(self, w):
        w *= self.mask_var
        return w

    def get_config(self):
        return {'mask': self.mask}

    def update(self, mask):
        self.mask = mask
        K.set_value(self.mask_var, mask)


def find_first_pos(array, value):
    idx = (np.abs(array - value)).argmin()
    return idx


def find_last_pos(array, value):
    idx = (np.abs(array - value))[::-1].argmin()
    return array.shape[0] - idx


def createWeightsMask(epsilon, no_rows, no_cols):
    # generate an Erdos Renyi sparse weights mask
    mask_weights = np.random.rand(no_rows, no_cols)
    prob = 1 - (epsilon * (no_rows + no_cols)) / (
        no_rows * no_cols)  # normal tp have 8x connections
    mask_weights[mask_weights < prob] = 0
    mask_weights[mask_weights >= prob] = 1
    no_parameters = np.sum(mask_weights)
    print(f"Create Sparse Matrix: {no_parameters=}, {no_rows=}, {no_cols=}")
    return [no_parameters, mask_weights]


class SET_MLP_CIFAR10:

    def __init__(self, strategy: BaseSETStrategy, max_epochs):
        self.strategy = strategy

        # set model parameters
        self.epsilon = 20  # control the sparsity level as discussed in the paper
        self.zeta = 0.3  # the fraction of the weights removed
        self.batch_size = 100  # batch size
        self.maxepoches = max_epochs  # number of epochs
        self.learning_rate = 0.01  # SGD learning rate
        self.num_classes = 10  # number of classes
        self.momentum = 0.9  # SGD momentum

        # generate an Erdos Renyi sparse weights mask for each layer
        [self.noPar1,
         self.wm1_buffer] = createWeightsMask(self.epsilon, 32 * 32 * 3, 4000)
        [self.noPar2,
         self.wm2_buffer] = createWeightsMask(self.epsilon, 4000, 1000)
        [self.noPar3,
         self.wm3_buffer] = createWeightsMask(self.epsilon, 1000, 4000)

        # initialize layers weights
        self.w1 = None
        self.w2 = None
        self.w3 = None
        self.w4 = None

        # initialize weights for SReLu activation function
        self.wSRelu1 = None
        self.wSRelu2 = None
        self.wSRelu3 = None

        self.wm1_core_buffer = np.zeros_like(self.wm1_buffer, dtype=np.float32)

        self.wm2_core_buffer = np.zeros_like(self.wm2_buffer, dtype=np.float32)

        self.wm3_core_buffer = np.zeros_like(self.wm3_buffer, dtype=np.float32)

        # create a SET-MLP model
        self.create_model()

    def create_model(self):

        # create a SET-MLP model for CIFAR10 with 3 hidden layers
        self.model = Sequential()
        self.model.add(Flatten(input_shape=(32, 32, 3)))
        self.model.add(
            Dense(4000,
                  name="sparse_1",
                  kernel_constraint=MaskWeights(self.wm1_buffer)))
        self.model.add(SReLU(name="srelu1"))
        self.model.add(Dropout(0.3))
        self.model.add(
            Dense(1000,
                  name="sparse_2",
                  kernel_constraint=MaskWeights(self.wm2_buffer)))
        self.model.add(SReLU(name="srelu2"))
        self.model.add(Dropout(0.3))
        self.model.add(
            Dense(4000,
                  name="sparse_3",
                  kernel_constraint=MaskWeights(self.wm3_buffer)))
        self.model.add(SReLU(name="srelu3"))
        self.model.add(Dropout(0.3))
        self.model.add(
            Dense(self.num_classes, name="dense_4")
        )  #please note that there is no need for a sparse output layer as the number of classes is much smaller than the number of input hidden neurons
        self.model.add(Activation('softmax'))

        # If we already have weights from a previous training phase, reapply them
        self._restore_previous_weights()

    def _restore_previous_weights(self):
        # helper to load previously stored weights into freshly built layers
        def maybe_set(name, weights):
            if weights is not None:
                self.model.get_layer(name).set_weights(weights)

        maybe_set("sparse_1", self.w1)
        maybe_set("srelu1", self.wSRelu1)
        maybe_set("sparse_2", self.w2)
        maybe_set("srelu2", self.wSRelu2)
        maybe_set("sparse_3", self.w3)
        maybe_set("srelu3", self.wSRelu3)
        maybe_set("dense_4", self.w4)

    def rewireMask(self,
                   weights,
                   no_weights,
                   mask,
                   core_buffer,
                   mask_buffer,
                   extra_info=None):

        param_id = id(mask_buffer)

        # remove zeta largest negative and smallest positive weights
        mask_buffer = self.strategy.prune_neurons(mask_buffer, weights.ravel(),
                                                  mask.ravel(), extra_info)

        assert id(mask_buffer) == param_id, "Prune related in rebind"

        assert mask_buffer.shape == weights.shape

        np.copyto(core_buffer, mask_buffer)

        # occupied = set(zip(*np.where(rewiredWeights == 1)))
        no_rewires = int(no_weights - np.sum(mask_buffer))

        assert no_rewires > 0, "If no weights are rewired, there likely is a bug your code!"

        self.strategy.regrow_neurons(no_rewires, weights.shape, mask_buffer,
                                     extra_info)

        # for i, j in new_positions:
        #     rewiredWeights[i, j] = 1

        # noRewires = noWeights - np.sum(rewiredWeights)
        # zero_positions = np.argwhere(rewiredWeights == 0)
        # indices_to_add = np.random.choice(len(zero_positions),
        #                                   size=int(
        #                                       min(noRewires,
        #                                           len(zero_positions))),
        #                                   replace=False)
        # positions_to_add = zero_positions[indices_to_add]
        #
        # rewiredWeights[positions_to_add[:, 0], positions_to_add[:, 1]] = 1

        return [mask_buffer, core_buffer]

    def weightsEvolution(self):
        # this represents the core of the SET procedure. It removes the weights closest to zero in each layer and add new random weights
        self.w1 = self.model.get_layer("sparse_1").get_weights()
        self.w2 = self.model.get_layer("sparse_2").get_weights()
        self.w3 = self.model.get_layer("sparse_3").get_weights()
        self.w4 = self.model.get_layer("dense_4").get_weights()

        self.wSRelu1 = self.model.get_layer("srelu1").get_weights()
        self.wSRelu2 = self.model.get_layer("srelu2").get_weights()
        self.wSRelu3 = self.model.get_layer("srelu3").get_weights()

        global old_weights

        if old_weights is not None:
            print(f"equal? {zero_pattern_equal(old_weights, self.w1[0])}")

        old_weights = self.w1[0]

        match self.strategy.__class__.__name__:
            case "RandomSET":
                [self.wm1_buffer, self.wm1_core_buffer
                 ] = self.rewireMask(self.w1[0], self.noPar1, self.wm1_buffer,
                                     self.wm1_core_buffer, self.wm1_buffer,
                                     {"temp_buf": self.wm1_core_buffer})
                [self.wm2_buffer, self.wm2_core_buffer
                 ] = self.rewireMask(self.w2[0], self.noPar2, self.wm2_buffer,
                                     self.wm2_core_buffer, self.wm2_buffer,
                                     {"temp_buf": self.wm2_core_buffer})
                [self.wm3_buffer, self.wm3_core_buffer
                 ] = self.rewireMask(self.w3[0], self.noPar3, self.wm3_buffer,
                                     self.wm3_core_buffer, self.wm3_buffer,
                                     {"temp_buf": self.wm3_core_buffer})
            case "NeuronCentralitySET":
                [self.wm1_buffer, self.wm1_core_buffer
                 ] = self.rewireMask(self.w1[0], self.noPar1, self.wm1_buffer,
                                     self.wm1_core_buffer, self.wm1_buffer, {
                                         "layer": "layer_1",
                                         "self": self
                                     })
                [self.wm2_buffer, self.wm2_core_buffer
                 ] = self.rewireMask(self.w2[0], self.noPar2, self.wm2_buffer,
                                     self.wm2_core_buffer, self.wm2_buffer, {
                                         "layer": "layer_2",
                                         "self": self
                                     })
                [self.wm3_buffer, self.wm3_core_buffer
                 ] = self.rewireMask(self.w3[0], self.noPar3, self.wm3_buffer,
                                     self.wm3_core_buffer, self.wm3_buffer, {
                                         "layer": "layer_3",
                                         "self": self
                                     })
            case _:
                raise NotImplementedError(
                    f"Strategy {self.strategy.__class__.__name__} not implemented"
                )

        self.w1[0] = self.w1[0] * self.wm1_buffer
        self.w2[0] = self.w2[0] * self.wm2_buffer
        self.w3[0] = self.w3[0] * self.wm3_buffer

        self.model.get_layer("sparse_1").kernel_constraint.update(
            self.wm1_buffer)
        self.model.get_layer("sparse_2").kernel_constraint.update(
            self.wm2_buffer)
        self.model.get_layer("sparse_3").kernel_constraint.update(
            self.wm3_buffer)

    def train(self, target_accuracy=1.0):

        # read CIFAR10 data
        [x_train, x_test, y_train, y_test] = self.read_data()

        steps_per_epoch = x_train.shape[0] // self.batch_size

        # #data augmentation
        # datagen = ImageDataGenerator(
        #     featurewise_center=False,  # set input mean to 0 over the dataset
        #     samplewise_center=False,  # set each sample mean to 0
        #     featurewise_std_normalization=
        #     False,  # divide inputs by std of the dataset
        #     samplewise_std_normalization=False,  # divide each input by its std
        #     zca_whitening=False,  # apply ZCA whitening
        #     rotation_range=
        #     10,  # randomly rotate images in the range (degrees, 0 to 180)
        #     width_shift_range=
        #     0.1,  # randomly shift images horizontally (fraction of total width)
        #     height_shift_range=
        #     0.1,  # randomly shift images vertically (fraction of total height)
        #     horizontal_flip=True,  # randomly flip images
        #     vertical_flip=False)  # randomly flip images
        # datagen.fit(x_train)

        data_augmentation = tf.keras.Sequential(
            [
                tf.keras.layers.RandomFlip("horizontal"),
                tf.keras.layers.RandomTranslation(0.1, 0.1),
                tf.keras.layers.RandomRotation(10 / 360.0),  # 10 degrees
            ],
            name="data_augmentation")

        train_ds = tf.data.Dataset.from_tensor_slices((x_train, y_train))
        # train_ds = train_ds.shuffle(50000, reshuffle_each_iteration=True)
        train_ds = train_ds.batch(self.batch_size)
        train_ds = train_ds.map(lambda x, y:
                                (data_augmentation(x, training=True), y),
                                num_parallel_calls=AUTOTUNE)
        train_ds = train_ds.prefetch(AUTOTUNE)

        val_ds = tf.data.Dataset.from_tensor_slices((x_test, y_test))
        val_ds = val_ds.batch(self.batch_size).prefetch(AUTOTUNE)

        self.model.summary()

        # training process in a for loop
        self.accuracies_per_epoch = []
        epoch_count = -1
        best_accuracy = 0.0

        sgd = optimizers.SGD(learning_rate=self.learning_rate,
                             momentum=self.momentum)
        self.model.compile(loss='categorical_crossentropy',
                           optimizer=sgd,
                           metrics=['accuracy'])

        for epoch in range(0, self.maxepoches):

            historytemp = self.model.fit(train_ds,
                                         steps_per_epoch=steps_per_epoch,
                                         epochs=epoch,
                                         validation_data=val_ds,
                                         initial_epoch=epoch - 1)

            accuracy = historytemp.history['val_accuracy'][0]
            if accuracy > best_accuracy:
                best_accuracy = accuracy
            self.accuracies_per_epoch.append(accuracy)

            if accuracy >= target_accuracy:
                epoch_count = epoch
                break

            #ugly hack to avoid tensorflow memory increase for multiple fit_generator calls. Theano shall work more nicely this but it is outdated in general
            self.weightsEvolution()
            # K.clear_session()
            # self.create_model()
            self._restore_previous_weights()

        self.accuracies_per_epoch = np.asarray(self.accuracies_per_epoch)
        return epoch_count, best_accuracy

    def read_data(self):

        #read CIFAR10 data
        (x_train, y_train), (x_test, y_test) = cifar10.load_data()
        y_train = to_categorical(y_train, self.num_classes)
        y_test = to_categorical(y_test, self.num_classes)
        x_train = x_train.astype('float32')
        x_test = x_test.astype('float32')

        #normalize data
        x_train_mean = np.mean(x_train, axis=0)
        x_train_std = np.std(x_train, axis=0)
        x_train = (x_train - x_train_mean) / x_train_std
        x_test = (x_test - x_train_mean) / x_train_std

        return [x_train, x_test, y_train, y_test]


if __name__ == '__main__':
    set_strategy = RandomSET()

    # create and run a SET-MLP model on CIFAR10
    model = SET_MLP_CIFAR10(set_strategy, max_epochs=60)

    # train the SET-MLP model until 40%
    epoch_count = model.train(target_accuracy=0.4)
    print(f"took {epoch_count} epochs until convergance")

    # save accuracies over for all training epochs
    # in "results" folder you can find the output of running this file
    np.savetxt("results/set_mlp_srelu_sgd_cifar10_acc.txt",
               np.asarray(model.accuracies_per_epoch))
