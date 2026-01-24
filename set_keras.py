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

from __future__ import division, print_function

import inspect
import numpy as np
import tensorflow as tf
from tensorflow.keras import backend as K

from keras.models import Sequential, Model
from keras.layers import Dense, Dropout, Activation, Flatten, Input, Add
from keras import optimizers
from keras.constraints import Constraint
from keras.datasets import cifar10
from keras.utils import to_categorical

from strategies.base_strategy import BaseSETStrategy
from srelu import SReLU

from strategies.random_set import RandomSET
from strategies.neuron_centrality import NeuronCentralitySET
from strategies.ema import NeuronEMASet
from strategies.fisher_diagonal_set import FisherDiagonalSET
from strategies.fisher_diagonal_skip_set import FisherDiagonalSkipSET
from strategies.neuron_centrality_skip import NeuronCentralitySkipSET
from strategies.ema_skip import NeuronEMASkipSET

AUTOTUNE = tf.data.AUTOTUNE


def createWeightsMask(epsilon, noRows, noCols):
    mask_weights = np.random.rand(noRows, noCols)
    prob = 1 - (epsilon * (noRows + noCols)) / (noRows * noCols)
    mask_weights[mask_weights < prob] = 0
    mask_weights[mask_weights >= prob] = 1
    noParameters = np.sum(mask_weights)
    print("Create Sparse Matrix: No parameters, NoRows, NoCols ", noParameters,
          noRows, noCols)
    return [noParameters, mask_weights.astype(np.float32)]


class MaskWeights(Constraint):
    """
    Mask constraint that supports runtime updates via .update(new_mask).
    """

    def __init__(self, mask):
        self.mask = mask.astype(np.float32, copy=False)
        self.mask_var = K.variable(self.mask, K.floatx())

    def __call__(self, w):
        w *= self.mask_var
        return w

    def get_config(self):
        return {'mask': self.mask}

    def update(self, mask):
        self.mask = mask.astype(np.float32, copy=False)
        K.set_value(self.mask_var, self.mask)


class SET_MLP_CIFAR10:

    def __init__(self, strategy: BaseSETStrategy, max_epochs):
        if inspect.isclass(strategy):
            raise TypeError(
                f"Expected strategy instance, got class {strategy.__name__}. Did you forget ()?"
            )
        self.strategy = strategy

        self.epsilon = 20
        self.zeta = 0.3
        self.batch_size = 100
        self.maxepoches = max_epochs
        self.learning_rate = 0.01
        self.num_classes = 10
        self.momentum = 0.9

        # masks main buffers
        [self.noPar1,
         self.wm1_buffer] = createWeightsMask(self.epsilon, 32 * 32 * 3, 4000)
        [self.noPar2,
         self.wm2_buffer] = createWeightsMask(self.epsilon, 4000, 1000)
        [self.noPar3,
         self.wm3_buffer] = createWeightsMask(self.epsilon, 1000, 4000)

        # core buffers
        self.wm1_core_buffer = np.zeros_like(self.wm1_buffer, dtype=np.float32)
        self.wm2_core_buffer = np.zeros_like(self.wm2_buffer, dtype=np.float32)
        self.wm3_core_buffer = np.zeros_like(self.wm3_buffer, dtype=np.float32)

        self.noParSkip02 = None
        self.wmSkip02_buffer = None
        self.wmSkip02_core_buffer = None

        # weights holders
        self.w1 = self.w2 = self.w3 = self.w4 = None
        self.wSkip02 = None

        # SReLU weights
        self.wSRelu1 = self.wSRelu2 = self.wSRelu3 = None

        self.create_model()

    def create_model(self):

        use_skip = isinstance(
            self.strategy,
            (FisherDiagonalSkipSET, NeuronCentralitySkipSET, NeuronEMASkipSET))

        # allocate skip masks if needed (once)
        if use_skip and self.wmSkip02_buffer is None:
            [self.noParSkip02,
             self.wmSkip02_buffer] = createWeightsMask(self.epsilon,
                                                       32 * 32 * 3, 1000)
            self.wmSkip02_core_buffer = np.zeros_like(self.wmSkip02_buffer,
                                                      dtype=np.float32)

        if not use_skip:
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
            self.model.add(Dense(self.num_classes, name="dense_4"))
            self.model.add(Activation("softmax"))
            self._restore_previous_weights()
            return

        # skip model (Functional): input -> add into layer_2
        inp = Input(shape=(32, 32, 3))
        x = Flatten(name="flatten")(inp)

        h1 = Dense(4000,
                   name="sparse_1",
                   kernel_constraint=MaskWeights(self.wm1_buffer))(x)
        h1 = SReLU(name="srelu1")(h1)
        h1 = Dropout(0.3)(h1)

        h2_main = Dense(1000,
                        name="sparse_2",
                        kernel_constraint=MaskWeights(self.wm2_buffer))(h1)
        h2_skip = Dense(1000,
                        name="skip_02",
                        kernel_constraint=MaskWeights(self.wmSkip02_buffer))(x)
        h2 = Add(name="add_02")([h2_main, h2_skip])
        h2 = SReLU(name="srelu2")(h2)
        h2 = Dropout(0.3)(h2)

        h3 = Dense(4000,
                   name="sparse_3",
                   kernel_constraint=MaskWeights(self.wm3_buffer))(h2)
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

        maybe_set("skip_02", self.wSkip02)

        maybe_set("sparse_3", self.w3)
        maybe_set("srelu3", self.wSRelu3)

        maybe_set("dense_4", self.w4)

    def compute_kernel_grads_no_skip(self, x_batch, y_batch):
        k1 = self.model.get_layer("sparse_1").kernel
        k2 = self.model.get_layer("sparse_2").kernel
        k3 = self.model.get_layer("sparse_3").kernel

        loss_fn = tf.keras.losses.CategoricalCrossentropy()
        with tf.GradientTape() as tape:
            y_pred = self.model(x_batch, training=True)
            loss = loss_fn(y_batch, y_pred)

        g1, g2, g3 = tape.gradient(loss, [k1, k2, k3])
        return [g1.numpy(), g2.numpy(), g3.numpy()]

    def compute_kernel_grads_with_skip(self, x_batch, y_batch):
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

    def _call_prune(self, mask_buffer, weights_flat, mask_flat, extra_info):
        fn = self.strategy.prune_neurons
        for attempt in (
                lambda: fn(mask_buffer, weights_flat, mask_flat, extra_info),
                lambda: fn(mask_buffer, weights_flat, mask_flat),
                lambda: fn(mask_buffer, weights_flat, extra_info),
                lambda: fn(mask_buffer, weights_flat),
        ):
            try:
                return attempt()
            except TypeError:
                continue
        try:
            return fn(mask_buffer=mask_buffer,
                      weight_values=weights_flat,
                      weight_positions=mask_flat,
                      extra_info=extra_info)
        except TypeError:
            pass
        raise TypeError(
            f"Could not call prune_neurons() for {self.strategy.__class__.__name__}"
        )

    def _call_regrow(self, noRewires, shape, mask_buffer, extra_info):
        fn = self.strategy.regrow_neurons
        for attempt in (
                lambda: fn(noRewires, shape, mask_buffer, extra_info),
                lambda: fn(noRewires, shape, mask_buffer),
        ):
            try:
                return attempt()
            except TypeError:
                continue
        try:
            return fn(num_to_add=noRewires,
                      dimensions=shape,
                      mask=mask_buffer,
                      extra_info=extra_info)
        except TypeError:
            pass
        raise TypeError(
            f"Could not call regrow_neurons() for {self.strategy.__class__.__name__}"
        )

    def rewireMask(self,
                   weights,
                   noWeights,
                   mask_buffer,
                   core_buffer,
                   extra_info=None,
                   fisher_g=None,
                   fisher_v=None):
        """
        BUFFER MINDSET:
          1) prune -> writes into mask_buffer
          2) core_buffer snapshots post-prune mask
          3) regrow -> modifies mask_buffer again
        """
        prune_extra = {} if extra_info is None else dict(extra_info)
        if fisher_v is not None:
            prune_extra["v_flat"] = fisher_v.ravel()

        out = self._call_prune(mask_buffer, weights.ravel(),
                               mask_buffer.ravel(), prune_extra)
        if out is not None and out is not mask_buffer:
            np.copyto(mask_buffer, out)

        # snapshot core right after prune
        np.copyto(core_buffer, mask_buffer)

        noRewires = int(noWeights - np.sum(mask_buffer))
        if noRewires <= 0:
            return mask_buffer, core_buffer

        grow_extra = {} if extra_info is None else dict(extra_info)
        if fisher_g is not None and fisher_v is not None:
            grow_extra["g"] = fisher_g
            grow_extra["v"] = fisher_v
            grow_extra["mask"] = (mask_buffer == 1)

        self._call_regrow(noRewires, weights.shape, mask_buffer, grow_extra)
        return mask_buffer, core_buffer

    def _prune_only_into_buffers(self,
                                 weights,
                                 noWeights,
                                 mask_buffer,
                                 core_buffer,
                                 extra_info=None,
                                 fisher_v=None):
        """
        Like rewireMask but WITHOUT regrow.
        Used for shared-budget chooser (W2 vs skip).
        """
        prune_extra = {} if extra_info is None else dict(extra_info)
        if fisher_v is not None:
            prune_extra["v_flat"] = fisher_v.ravel()

        out = self._call_prune(mask_buffer, weights.ravel(),
                               mask_buffer.ravel(), prune_extra)
        if out is not None and out is not mask_buffer:
            np.copyto(mask_buffer, out)

        np.copyto(core_buffer, mask_buffer)
        need = int(noWeights - np.sum(mask_buffer))
        return need

    def fisher_choose_between_W2_and_skip(self, g2, v2, gskip, vskip):
        """
        Prune W2 and skip separately (buffer-style), snapshot cores,
        then allocate regrowth budget across both using fisher grow score g^2/(v+eps).
        """
        eps = getattr(self.strategy, "eps", 1e-8)

        need2 = self._prune_only_into_buffers(weights=self.w2[0],
                                              noWeights=self.noPar2,
                                              mask_buffer=self.wm2_buffer,
                                              core_buffer=self.wm2_core_buffer,
                                              fisher_v=v2)
        needs = self._prune_only_into_buffers(
            weights=self.wSkip02[0],
            noWeights=self.noParSkip02,
            mask_buffer=self.wmSkip02_buffer,
            core_buffer=self.wmSkip02_core_buffer,
            fisher_v=vskip)

        total_need = need2 + needs
        if total_need <= 0:
            return

        inactive2 = np.argwhere(self.wm2_buffer == 0)
        inactives = np.argwhere(self.wmSkip02_buffer == 0)
        if inactive2.size == 0 and inactives.size == 0:
            return

        cand_total = min(total_need * 20,
                         inactive2.shape[0] + inactives.shape[0])
        cand2 = min(cand_total // 2, inactive2.shape[0])
        cands = min(cand_total - cand2, inactives.shape[0])

        merged = []

        if cand2 > 0:
            idx = np.random.choice(inactive2.shape[0],
                                   size=cand2,
                                   replace=False)
            c = inactive2[idx]
            ci, cj = c[:, 0], c[:, 1]
            score = (g2[ci, cj]**2) / (v2[ci, cj] + eps)
            for k in range(cand2):
                merged.append(("W2", int(ci[k]), int(cj[k]), float(score[k])))

        if cands > 0:
            idx = np.random.choice(inactives.shape[0],
                                   size=cands,
                                   replace=False)
            c = inactives[idx]
            ci, cj = c[:, 0], c[:, 1]
            score = (gskip[ci, cj]**2) / (vskip[ci, cj] + eps)
            for k in range(cands):
                merged.append(("WS", int(ci[k]), int(cj[k]), float(score[k])))

        merged.sort(key=lambda t: t[3], reverse=True)
        merged = merged[:total_need]

        for which, i, j, _ in merged:
            if which == "W2":
                self.wm2_buffer[i, j] = 1.0
            else:
                self.wmSkip02_buffer[i, j] = 1.0

    def weightsEvolution(self, fisher_payload=None):
        self.w1 = self.model.get_layer("sparse_1").get_weights()
        self.w2 = self.model.get_layer("sparse_2").get_weights()
        self.w3 = self.model.get_layer("sparse_3").get_weights()
        self.w4 = self.model.get_layer("dense_4").get_weights()

        self.wSRelu1 = self.model.get_layer("srelu1").get_weights()
        self.wSRelu2 = self.model.get_layer("srelu2").get_weights()
        self.wSRelu3 = self.model.get_layer("srelu3").get_weights()

        use_skip = isinstance(
            self.strategy,
            (FisherDiagonalSkipSET, NeuronCentralitySkipSET, NeuronEMASkipSET))
        if use_skip:
            self.wSkip02 = self.model.get_layer("skip_02").get_weights()

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

            case "NeuronEMASet":
                self.rewireMask(self.w1[0],
                                self.noPar1,
                                self.wm1_buffer,
                                self.wm1_core_buffer,
                                extra_info={
                                    "layer": "layer_1",
                                    "self": self
                                })
                self.rewireMask(self.w2[0],
                                self.noPar2,
                                self.wm2_buffer,
                                self.wm2_core_buffer,
                                extra_info={
                                    "layer": "layer_2",
                                    "self": self
                                })
                self.rewireMask(self.w3[0],
                                self.noPar3,
                                self.wm3_buffer,
                                self.wm3_core_buffer,
                                extra_info={
                                    "layer": "layer_3",
                                    "self": self
                                })

            case "FisherDiagonalSET":
                if fisher_payload is None:
                    raise ValueError(
                        "FisherDiagonalSET requires fisher_payload.")
                g1, g2, _, g3 = fisher_payload["grads"]
                v1, v2, _, v3 = fisher_payload["Vs"]

                self.rewireMask(self.w1[0],
                                self.noPar1,
                                self.wm1_buffer,
                                self.wm1_core_buffer,
                                fisher_g=g1,
                                fisher_v=v1)
                self.rewireMask(self.w2[0],
                                self.noPar2,
                                self.wm2_buffer,
                                self.wm2_core_buffer,
                                fisher_g=g2,
                                fisher_v=v2)
                self.rewireMask(self.w3[0],
                                self.noPar3,
                                self.wm3_buffer,
                                self.wm3_core_buffer,
                                fisher_g=g3,
                                fisher_v=v3)

            case "FisherDiagonalSkipSET":
                if fisher_payload is None:
                    raise ValueError(
                        "FisherDiagonalSkipSET requires fisher_payload.")
                g1, g2, gskip, g3 = fisher_payload["grads"]
                v1, v2, vskip, v3 = fisher_payload["Vs"]

                # W1 & W3 normal fisher
                self.rewireMask(self.w1[0],
                                self.noPar1,
                                self.wm1_buffer,
                                self.wm1_core_buffer,
                                fisher_g=g1,
                                fisher_v=v1)
                self.rewireMask(self.w3[0],
                                self.noPar3,
                                self.wm3_buffer,
                                self.wm3_core_buffer,
                                fisher_g=g3,
                                fisher_v=v3)

                # W2 vs skip shared-budget
                self.fisher_choose_between_W2_and_skip(g2=g2,
                                                       v2=v2,
                                                       gskip=gskip,
                                                       vskip=vskip)

            case "NeuronCentralitySkipSET":
                # W1 & W3 normal centrality
                self.rewireMask(self.w1[0],
                                self.noPar1,
                                self.wm1_buffer,
                                self.wm1_core_buffer,
                                extra_info={
                                    "layer": "layer_1",
                                    "self": self
                                })
                self.rewireMask(self.w3[0],
                                self.noPar3,
                                self.wm3_buffer,
                                self.wm3_core_buffer,
                                extra_info={
                                    "layer": "layer_3",
                                    "self": self
                                })

                # W2 vs skip shared-budget centrality chooser
                self.strategy.choose_between_w2_and_skip(self)
            case "NeuronEMASkipSET":
                # W1 & W3 normal EMA
                self.rewireMask(self.w1[0],
                                self.noPar1,
                                self.wm1_buffer,
                                self.wm1_core_buffer,
                                extra_info={
                                    "layer": "layer_1",
                                    "self": self
                                })
                self.rewireMask(self.w3[0],
                                self.noPar3,
                                self.wm3_buffer,
                                self.wm3_core_buffer,
                                extra_info={
                                    "layer": "layer_3",
                                    "self": self
                                })

                # W2 vs Skip shared-budget chooser
                self.strategy.choose_between_w2_and_skip(self)

            case _:
                raise NotImplementedError(
                    f"Strategy {self.strategy.__class__.__name__} not implemented"
                )

    def rewireMask(self, weights_2d, noWeights: int, mask_2d, core_buffer_2d,
                   extra_info: dict):
        # prune modifies mask_2d in-place
        self.strategy.prune_neurons(mask_2d, weights_2d.ravel(),
                                    mask_2d.ravel(), extra_info)

        np.copyto(core_buffer_2d, mask_2d)

        noRewires = int(noWeights - np.sum(mask_2d))
        if noRewires <= 0:
            return

        # regrow modifies mask_2d in-place
        self.strategy.regrow_neurons(noRewires, weights_2d.shape, mask_2d,
                                     extra_info)

    def weightsEvolution(self, fisher_payload: dict | None = None):
        self._capture_weights()

        # Convenience: common extra_info keys used by your in-place strategies
        def mk_extra(layer_name: str, core_buf: np.ndarray):
            # Some strategies (RandomSET in your repo) assert extra_info is not None,
            # and may use temp buffers. So always provide at least temp_buf.
            ex = {
                "layer": layer_name,
                "self": self,
                "temp_buf": core_buf,
            }
            return ex

        needs_grad = bool(getattr(self.strategy, "needs_gradients", False))

        #Layer 1
        if needs_grad:
            v1 = fisher_payload["Vs"][0]
            g1 = fisher_payload["grads"][0]
            ex1 = mk_extra("layer_1", self.wm1_core_buffer)
            ex1.update({"v_flat": v1.ravel(), "g": g1, "v": v1})
        else:
            ex1 = mk_extra("layer_1", self.wm1_core_buffer)

        self.rewireMask(self.w1[0], self.noPar1, self.wm1_buffer,
                        self.wm1_core_buffer, ex1)

        #Layer 2
        if needs_grad:
            if self.use_skip:
                v2 = fisher_payload["Vs"][1]
                g2 = fisher_payload["grads"][1]
            else:
                v2 = fisher_payload["Vs"][1]
                g2 = fisher_payload["grads"][1]
            ex2 = mk_extra("layer_2", self.wm2_core_buffer)
            ex2.update({"v_flat": v2.ravel(), "g": g2, "v": v2})
        else:
            ex2 = mk_extra("layer_2", self.wm2_core_buffer)

        self.rewireMask(self.w2[0], self.noPar2, self.wm2_buffer,
                        self.wm2_core_buffer, ex2)

        #Layer 3
        if needs_grad:
            if self.use_skip:
                v3 = fisher_payload["Vs"][-1]
                g3 = fisher_payload["grads"][-1]
            else:
                v3 = fisher_payload["Vs"][2]
                g3 = fisher_payload["grads"][2]
            ex3 = mk_extra("layer_3", self.wm3_core_buffer)
            ex3.update({"v_flat": v3.ravel(), "g": g3, "v": v3})
        else:
            ex3 = mk_extra("layer_3", self.wm3_core_buffer)

        self.rewireMask(self.w3[0], self.noPar3, self.wm3_buffer,
                        self.wm3_core_buffer, ex3)

        #Skip layer
        if self.use_skip:
            if needs_grad:

                vskip = fisher_payload["Vs"][2]
                gskip = fisher_payload["grads"][2]
                exs = mk_extra("skip_02", self.wmSkip02_core_buffer)
                exs.update({"v_flat": vskip.ravel(), "g": gskip, "v": vskip})
            else:
                exs = mk_extra("skip_02", self.wmSkip02_core_buffer)

            self.rewireMask(
                self.wSkip02[0],
                self.noParSkip02,
                self.wmSkip02_buffer,
                self.wmSkip02_core_buffer,
                exs,
            )

        # apply core masks to weights
        self.w1[0] *= self.wm1_buffer
        self.w2[0] *= self.wm2_buffer
        self.w3[0] *= self.wm3_buffer
        if self.use_skip:
            self.wSkip02[0] *= self.wmSkip02_buffer

        self.model.get_layer("sparse_1").kernel_constraint.update(
            self.wm1_buffer)
        self.model.get_layer("sparse_2").kernel_constraint.update(
            self.wm2_buffer)
        self.model.get_layer("sparse_3").kernel_constraint.update(
            self.wm3_buffer)
        if use_skip:
            self.model.get_layer("skip_02").kernel_constraint.update(
                self.wmSkip02_buffer)

    def train(self, target_accuracy=1.0):
        [x_train, x_test, y_train, y_test] = self.read_data()
<<<<<<< HEAD
=======
        self._ema_x_train = x_train  # EMA strategy uses this
>>>>>>> d3cea24 (Update set_keras.py)

        steps_per_epoch = x_train.shape[0] // self.batch_size

        data_augmentation = tf.keras.Sequential(
            [
                tf.keras.layers.RandomFlip("horizontal"),
                tf.keras.layers.RandomTranslation(0.1, 0.1),
                tf.keras.layers.RandomRotation(10 / 360.0),
            ],
            name="data_augmentation",
        )

        train_ds = tf.data.Dataset.from_tensor_slices((x_train, y_train))
        train_ds = train_ds.batch(self.batch_size)
        train_ds = train_ds.map(lambda x, y:
                                (data_augmentation(x, training=True), y),
                                num_parallel_calls=AUTOTUNE)
        train_ds = train_ds.prefetch(AUTOTUNE)

        val_ds = tf.data.Dataset.from_tensor_slices((x_test, y_test))
        val_ds = val_ds.batch(self.batch_size).prefetch(AUTOTUNE)

        self.model.summary()

        self.accuracies_per_epoch = []
        epoch_count = -1
        best_accuracy = 0.0

        sgd = optimizers.SGD(learning_rate=self.learning_rate,
                             momentum=self.momentum)
        self.model.compile(loss='categorical_crossentropy',
                           optimizer=sgd,
                           metrics=['accuracy'])

        # init V for fisher strategies
        if isinstance(self.strategy, FisherDiagonalSET):
            self.strategy.v1 = np.zeros_like(self.wm1_buffer, dtype=np.float32)
            self.strategy.v2 = np.zeros_like(self.wm2_buffer, dtype=np.float32)
            self.strategy.v3 = np.zeros_like(self.wm3_buffer, dtype=np.float32)

        if isinstance(self.strategy, FisherDiagonalSkipSET):
            self.strategy.v1 = np.zeros_like(self.wm1_buffer, dtype=np.float32)
            self.strategy.v2 = np.zeros_like(self.wm2_buffer, dtype=np.float32)
            self.strategy.v3 = np.zeros_like(self.wm3_buffer, dtype=np.float32)
            self.strategy.vSkip02 = np.zeros_like(self.wmSkip02_buffer,
                                                  dtype=np.float32)

        for epoch in range(0, self.maxepoches):
            historytemp = self.model.fit(train_ds,
                                         steps_per_epoch=steps_per_epoch,
                                         epochs=epoch,
                                         validation_data=val_ds,
                                         initial_epoch=epoch - 1,
                                         verbose=1)

            accuracy = historytemp.history['val_accuracy'][0]
            best_accuracy = max(best_accuracy, accuracy)
            self.accuracies_per_epoch.append(accuracy)

            if accuracy >= target_accuracy:
                epoch_count = epoch
                break

            fisher_payload = None

            if isinstance(self.strategy, FisherDiagonalSET):
                bs = 256
                g1, g2, g3 = self.compute_kernel_grads_no_skip(
                    x_train[:bs], y_train[:bs])
                b = self.strategy.beta
                self.strategy.v1 = b * self.strategy.v1 + (1.0 - b) * (
                    g1.astype(np.float32)**2)
                self.strategy.v2 = b * self.strategy.v2 + (1.0 - b) * (
                    g2.astype(np.float32)**2)
                self.strategy.v3 = b * self.strategy.v3 + (1.0 - b) * (
                    g3.astype(np.float32)**2)
                fisher_payload = {
                    "grads": (g1, g2, None, g3),
                    "Vs": (self.strategy.v1, self.strategy.v2, None,
                           self.strategy.v3)
                }

            if isinstance(self.strategy, FisherDiagonalSkipSET):
                bs = 256
                g1, g2, gskip, g3 = self.compute_kernel_grads_with_skip(
                    x_train[:bs], y_train[:bs])
                b = self.strategy.beta
                self.strategy.v1 = b * self.strategy.v1 + (1.0 - b) * (
                    g1.astype(np.float32)**2)
                self.strategy.v2 = b * self.strategy.v2 + (1.0 - b) * (
                    g2.astype(np.float32)**2)
                self.strategy.vSkip02 = b * self.strategy.vSkip02 + (
                    1.0 - b) * (gskip.astype(np.float32)**2)
                self.strategy.v3 = b * self.strategy.v3 + (1.0 - b) * (
                    g3.astype(np.float32)**2)
                fisher_payload = {
                    "grads": (g1, g2, gskip, g3),
                    "Vs": (self.strategy.v1, self.strategy.v2,
                           self.strategy.vSkip02, self.strategy.v3)
                }

            self.weightsEvolution(fisher_payload=fisher_payload)
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
