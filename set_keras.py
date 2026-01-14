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
# Author: Decebal Constantin Mocanu et al.;
# Proof of concept implementation of Sparse Evolutionary Training (SET) of Multi Layer Perceptron (MLP) on CIFAR10 using Keras and a mask over weights.


from __future__ import division
from __future__ import print_function

from tensorflow.keras.preprocessing.image import ImageDataGenerator
from keras.models import Sequential, Model
from keras.layers import Dense, Dropout, Activation, Flatten, Input, Add
from keras import optimizers
import numpy as np
from tensorflow.keras import backend as K
import tensorflow as tf

from strategies.base_strategy import BaseSETStrategy
from srelu import SReLU
from keras.datasets import cifar10
from keras.utils import to_categorical
from keras.constraints import Constraint

# strategies live in their own files
from strategies.random_set import RandomSET
from strategies.neuron_centrality import NeuronCentralitySET
from strategies.ema import NeuronEMASet
from strategies.fisher_diagonal_skip_set import FisherDiagonalSkipSET


class MaskWeights(Constraint):
    def __init__(self, mask):
        self.mask = mask
        self.mask = K.cast(self.mask, K.floatx())

    def __call__(self, w):
        w *= self.mask
        return w

    def get_config(self):
        return {'mask': self.mask}


def createWeightsMask(epsilon, noRows, noCols):
    mask_weights = np.random.rand(noRows, noCols)
    prob = 1 - (epsilon * (noRows + noCols)) / (noRows * noCols)
    mask_weights[mask_weights < prob] = 0
    mask_weights[mask_weights >= prob] = 1
    noParameters = np.sum(mask_weights)
    print("Create Sparse Matrix: No parameters, NoRows, NoCols ", noParameters, noRows, noCols)
    return [noParameters, mask_weights]


class SET_MLP_CIFAR10:
    def __init__(self, strategy: BaseSETStrategy, max_epochs):
        self.strategy = strategy

        self.epsilon = 20
        self.zeta = 0.3
        self.batch_size = 100
        self.maxepoches = max_epochs
        self.learning_rate = 0.01
        self.num_classes = 10
        self.momentum = 0.9

        # masks
        [self.noPar1, self.wm1] = createWeightsMask(self.epsilon, 32 * 32 * 3, 4000)
        [self.noPar2, self.wm2] = createWeightsMask(self.epsilon, 4000, 1000)
        [self.noPar3, self.wm3] = createWeightsMask(self.epsilon, 1000, 4000)

        # skip mask exists; only used when strategy is FisherDiagonalSkipSET
        [self.noParSkip02, self.wmSkip02] = createWeightsMask(self.epsilon, 32 * 32 * 3, 1000)

        # weights holders
        self.w1 = self.w2 = self.w3 = self.w4 = None
        self.wSkip02 = None

        # SReLU weights
        self.wSRelu1 = self.wSRelu2 = self.wSRelu3 = None

        self.create_model()

    def create_model(self):
        use_skip = isinstance(self.strategy, FisherDiagonalSkipSET)

        if not use_skip:
            # Sequential original (Random/Centrality/EMA)
            self.model = Sequential()
            self.model.add(Flatten(input_shape=(32, 32, 3)))
            self.model.add(Dense(4000, name="sparse_1", kernel_constraint=MaskWeights(self.wm1)))
            self.model.add(SReLU(name="srelu1"))
            self.model.add(Dropout(0.3))
            self.model.add(Dense(1000, name="sparse_2", kernel_constraint=MaskWeights(self.wm2)))
            self.model.add(SReLU(name="srelu2"))
            self.model.add(Dropout(0.3))
            self.model.add(Dense(4000, name="sparse_3", kernel_constraint=MaskWeights(self.wm3)))
            self.model.add(SReLU(name="srelu3"))
            self.model.add(Dropout(0.3))
            self.model.add(Dense(self.num_classes, name="dense_4"))
            self.model.add(Activation('softmax'))
            self._restore_previous_weights()
            return

        # FisherDiagonalSkipSET -> Functional model with skip: x -> layer2
        inp = Input(shape=(32, 32, 3))
        x = Flatten(name="flatten")(inp)

        h1 = Dense(4000, name="sparse_1", kernel_constraint=MaskWeights(self.wm1))(x)
        h1 = SReLU(name="srelu1")(h1)
        h1 = Dropout(0.3)(h1)

        h2_main = Dense(1000, name="sparse_2", kernel_constraint=MaskWeights(self.wm2))(h1)
        h2_skip = Dense(1000, name="skip_02", kernel_constraint=MaskWeights(self.wmSkip02))(x)
        h2 = Add(name="add_02")([h2_main, h2_skip])
        h2 = SReLU(name="srelu2")(h2)
        h2 = Dropout(0.3)(h2)

        h3 = Dense(4000, name="sparse_3", kernel_constraint=MaskWeights(self.wm3))(h2)
        h3 = SReLU(name="srelu3")(h3)
        h3 = Dropout(0.3)(h3)

        out = Dense(self.num_classes, name="dense_4")(h3)
        out = Activation("softmax")(out)

        self.model = Model(inputs=inp, outputs=out)
        self._restore_previous_weights()

    def _restore_previous_weights(self):
        names = set(l.name for l in self.model.layers)

        def maybe_set(name, weights):
            if weights is not None and name in names:
                self.model.get_layer(name).set_weights(weights)

        maybe_set("sparse_1", self.w1)
        maybe_set("srelu1", self.wSRelu1)

        maybe_set("sparse_2", self.w2)
        maybe_set("srelu2", self.wSRelu2)

        maybe_set("skip_02", self.wSkip02)  # only exists in skip model

        maybe_set("sparse_3", self.w3)
        maybe_set("srelu3", self.wSRelu3)

        maybe_set("dense_4", self.w4)

    # fisher gradient computation
    
    def compute_kernel_grads(self, x_batch, y_batch):
        k1 = self.model.get_layer("sparse_1").kernel
        k2 = self.model.get_layer("sparse_2").kernel
        k3 = self.model.get_layer("sparse_3").kernel
        kskip = self.model.get_layer("skip_02").kernel

        loss_fn = tf.keras.losses.CategoricalCrossentropy()
        with tf.GradientTape() as tape:
            y_pred = self.model(x_batch, training=True)
            loss = loss_fn(y_batch, y_pred)

        g1, g2, gskip, g3 = tape.gradient(loss, [k1, k2, kskip, k3])
        return [g1.numpy(), g2.numpy(), gskip.numpy(), g3.numpy()]


    # rewire all strategies
    def rewireMask(self, weights, noWeights, mask, extra_info=None, fisher_v=None, fisher_g=None):
        # fisher prune needs v_flat
        prune_extra = None
        if fisher_v is not None:
            prune_extra = {"v_flat": fisher_v.ravel()}

        keep_mask = self.strategy.prune_neurons(weights.ravel(), mask.ravel(), extra_info=prune_extra)
        rewired = keep_mask.reshape(weights.shape).astype(float)
        core = rewired.copy()

        occupied = set(zip(*np.where(rewired == 1)))
        noRewires = int(noWeights - np.sum(rewired))

        grow_extra = extra_info if extra_info is not None else {}
        grow_extra = dict(grow_extra)

        # fisher regrow needs g,v,mask
        if fisher_g is not None and fisher_v is not None:
            grow_extra["g"] = fisher_g
            grow_extra["v"] = fisher_v
            grow_extra["mask"] = (rewired == 1)

        new_positions = self.strategy.regrow_neurons(noRewires, weights.shape, occupied, grow_extra)
        for i, j in new_positions:
            rewired[i, j] = 1

        return rewired, core

    
    # fisher special: choose regrowth between W2 and skip
    def fisher_choose_between_W2_and_skip(self, W2, K2, M2, g2, v2,
                                         Wskip, Kskip, Mskip, gskip, vskip):

        # prune W2
        keep2 = self.strategy.prune_neurons(W2.ravel(), M2.ravel(), extra_info={"v_flat": v2.ravel()})
        newM2 = keep2.reshape(W2.shape).astype(float)
        core2 = newM2.copy()
        mask2 = (newM2 == 1)
        need2 = int(K2 - np.sum(newM2))

        # prune Wskip
        keeps = self.strategy.prune_neurons(Wskip.ravel(), Mskip.ravel(), extra_info={"v_flat": vskip.ravel()})
        newMs = keeps.reshape(Wskip.shape).astype(float)
        cores = newMs.copy()
        masks = (newMs == 1)
        needs = int(Kskip - np.sum(newMs))

        total_need = need2 + needs
        if total_need <= 0:
            return newM2, core2, newMs, cores

        inactive2 = np.argwhere(~mask2)
        inactives = np.argwhere(~masks)
        if inactive2.shape[0] == 0 and inactives.shape[0] == 0:
            return newM2, core2, newMs, cores

        cand_total = min(total_need * 20, inactive2.shape[0] + inactives.shape[0])
        cand2 = min(cand_total // 2, inactive2.shape[0])
        cands = min(cand_total - cand2, inactives.shape[0])

        merged = []

        if cand2 > 0:
            idx = np.random.choice(inactive2.shape[0], size=cand2, replace=False)
            c = inactive2[idx]
            ci, cj = c[:, 0], c[:, 1]
            score = (g2[ci, cj] ** 2) / (v2[ci, cj] + self.strategy.eps)
            for k in range(cand2):
                merged.append(("W2", int(ci[k]), int(cj[k]), float(score[k])))

        if cands > 0:
            idx = np.random.choice(inactives.shape[0], size=cands, replace=False)
            c = inactives[idx]
            ci, cj = c[:, 0], c[:, 1]
            score = (gskip[ci, cj] ** 2) / (vskip[ci, cj] + self.strategy.eps)
            for k in range(cands):
                merged.append(("WS", int(ci[k]), int(cj[k]), float(score[k])))

        merged.sort(key=lambda t: t[3], reverse=True)
        merged = merged[:total_need]

        for which, i, j, _ in merged:
            if which == "W2":
                newM2[i, j] = 1
            else:
                newMs[i, j] = 1

        return newM2, core2, newMs, cores


    def weightsEvolution(self, fisher_payload=None):
        self.w1 = self.model.get_layer("sparse_1").get_weights()
        self.w2 = self.model.get_layer("sparse_2").get_weights()
        self.w3 = self.model.get_layer("sparse_3").get_weights()
        self.w4 = self.model.get_layer("dense_4").get_weights()

        self.wSRelu1 = self.model.get_layer("srelu1").get_weights()
        self.wSRelu2 = self.model.get_layer("srelu2").get_weights()
        self.wSRelu3 = self.model.get_layer("srelu3").get_weights()

        use_skip = isinstance(self.strategy, FisherDiagonalSkipSET)
        if use_skip:
            self.wSkip02 = self.model.get_layer("skip_02").get_weights()

        match self.strategy.__class__.__name__:
            case "RandomSET":
                self.wm1, self.wm1Core = self.rewireMask(self.w1[0], self.noPar1, self.wm1)
                self.wm2, self.wm2Core = self.rewireMask(self.w2[0], self.noPar2, self.wm2)
                self.wm3, self.wm3Core = self.rewireMask(self.w3[0], self.noPar3, self.wm3)

            case "NeuronCentralitySET":
                self.wm1, self.wm1Core = self.rewireMask(self.w1[0], self.noPar1, self.wm1, {"layer": "layer_1", "self": self})
                self.wm2, self.wm2Core = self.rewireMask(self.w2[0], self.noPar2, self.wm2, {"layer": "layer_2", "self": self})
                self.wm3, self.wm3Core = self.rewireMask(self.w3[0], self.noPar3, self.wm3, {"layer": "layer_3", "self": self})

            case "NeuronEMASet":
                self.wm1, self.wm1Core = self.rewireMask(self.w1[0], self.noPar1, self.wm1, {"layer": "layer_1", "self": self})
                self.wm2, self.wm2Core = self.rewireMask(self.w2[0], self.noPar2, self.wm2, {"layer": "layer_2", "self": self})
                self.wm3, self.wm3Core = self.rewireMask(self.w3[0], self.noPar3, self.wm3, {"layer": "layer_3", "self": self})

            case "FisherDiagonalSkipSET":
                if fisher_payload is None:
                    raise ValueError("FisherDiagonalSkipSET requires fisher_payload.")

                g1, g2, gskip, g3 = fisher_payload["grads"]
                v1, v2, vskip, v3 = fisher_payload["Vs"]

                # W1 and W3: normal fisher prune+grow
                self.wm1, self.wm1Core = self.rewireMask(self.w1[0], self.noPar1, self.wm1, fisher_v=v1, fisher_g=g1)
                self.wm3, self.wm3Core = self.rewireMask(self.w3[0], self.noPar3, self.wm3, fisher_v=v3, fisher_g=g3)

                # W2 vs skip: choose regrowth globally
                self.wm2, self.wm2Core, self.wmSkip02, self.wmSkip02Core = self.fisher_choose_between_W2_and_skip(
                    W2=self.w2[0], K2=self.noPar2, M2=self.wm2, g2=g2, v2=v2,
                    Wskip=self.wSkip02[0], Kskip=self.noParSkip02, Mskip=self.wmSkip02, gskip=gskip, vskip=vskip
                )

            case _:
                raise NotImplementedError(f"Strategy {self.strategy.__class__.__name__} not implemented")

        # apply core masks
        self.w1[0] = self.w1[0] * self.wm1Core
        self.w2[0] = self.w2[0] * self.wm2Core
        self.w3[0] = self.w3[0] * self.wm3Core

        if use_skip:
            self.wSkip02[0] = self.wSkip02[0] * self.wmSkip02Core

            S
    # train
    def train(self, target_accuracy=1.0):
        [x_train, x_test, y_train, y_test] = self.read_data()

        # keep this for EMA strategy
        self._ema_x_train = x_train

        datagen = ImageDataGenerator(
            featurewise_center=False,
            samplewise_center=False,
            featurewise_std_normalization=False,
            samplewise_std_normalization=False,
            zca_whitening=False,
            rotation_range=10,
            width_shift_range=0.1,
            height_shift_range=0.1,
            horizontal_flip=True,
            vertical_flip=False
        )
        datagen.fit(x_train)

        self.model.summary()

        self.accuracies_per_epoch = []
        epoch_count = -1
        best_accuracy = 0.0

        sgd = optimizers.SGD(learning_rate=self.learning_rate, momentum=self.momentum)
        self.model.compile(loss='categorical_crossentropy', optimizer=sgd, metrics=['accuracy'])

        # init V for fisher strategy (strategy owns v arrays)
        if isinstance(self.strategy, FisherDiagonalSkipSET):
            self.strategy.v1 = np.zeros_like(self.wm1, dtype=np.float32)
            self.strategy.v2 = np.zeros_like(self.wm2, dtype=np.float32)
            self.strategy.v3 = np.zeros_like(self.wm3, dtype=np.float32)
            self.strategy.vSkip02 = np.zeros_like(self.wmSkip02, dtype=np.float32)

        for epoch in range(0, self.maxepoches):
            historytemp = self.model.fit(
                datagen.flow(x_train, y_train, batch_size=self.batch_size),
                steps_per_epoch=x_train.shape[0] // self.batch_size,
                epochs=epoch,
                validation_data=(x_test, y_test),
                initial_epoch=epoch - 1
            )

            accuracy = historytemp.history['val_accuracy'][0]
            best_accuracy = max(best_accuracy, accuracy)
            self.accuracies_per_epoch.append(accuracy)

            if accuracy >= target_accuracy:
                epoch_count = epoch
                break

            fisher_payload = None

            if isinstance(self.strategy, FisherDiagonalSkipSET):
                bs = 256
                grads = self.compute_kernel_grads(x_train[:bs], y_train[:bs])
                g1, g2, gskip, g3 = grads

                b = self.strategy.beta
                self.strategy.v1 = b * self.strategy.v1 + (1.0 - b) * (g1.astype(np.float32) ** 2)
                self.strategy.v2 = b * self.strategy.v2 + (1.0 - b) * (g2.astype(np.float32) ** 2)
                self.strategy.vSkip02 = b * self.strategy.vSkip02 + (1.0 - b) * (gskip.astype(np.float32) ** 2)
                self.strategy.v3 = b * self.strategy.v3 + (1.0 - b) * (g3.astype(np.float32) ** 2)

                fisher_payload = {
                    "grads": (g1, g2, gskip, g3),
                    "Vs": (self.strategy.v1, self.strategy.v2, self.strategy.vSkip02, self.strategy.v3)
                }

            self.weightsEvolution(fisher_payload=fisher_payload)

            # masks are captured at layer build time -> fisher must rebuild model
            if isinstance(self.strategy, FisherDiagonalSkipSET):
                K.clear_session()
                self.create_model()
            else:
                self._restore_previous_weights()

        self.accuracies_per_epoch = np.asarray(self.accuracies_per_epoch)
        return epoch_count, best_accuracy

    def read_data(self):
        (x_train, y_train), (x_test, y_test) = cifar10.load_data()
        y_train = to_categorical(y_train, self.num_classes)
        y_test = to_categorical(y_test, self.num_classes)
        x_train = x_train.astype('float32')
        x_test = x_test.astype('float32')

        xTrainMean = np.mean(x_train, axis=0)
        xTtrainStd = np.std(x_train, axis=0)
        x_train = (x_train - xTrainMean) / xTtrainStd
        x_test = (x_test - xTrainMean) / xTtrainStd

        return [x_train, x_test, y_train, y_test]
