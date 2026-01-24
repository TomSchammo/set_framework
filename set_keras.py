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

AUTOTUNE = tf.data.AUTOTUNE


# Mask constraint (updatable)
class MaskWeights(Constraint):
    def __init__(self, mask: np.ndarray):
        self.mask = mask.astype(np.float32, copy=False)
        self.mask_var = K.variable(self.mask, K.floatx())

    def __call__(self, w):
        w *= self.mask_var
        return w

    def get_config(self):
        return {"mask": self.mask}

    def update(self, mask: np.ndarray):
        self.mask = mask.astype(np.float32, copy=False)
        K.set_value(self.mask_var, self.mask)


        
# Mask init (Erdos-Renyi)

def createWeightsMask(epsilon: float, noRows: int, noCols: int):
    mask_weights = np.random.rand(noRows, noCols)
    prob = 1.0 - (epsilon * (noRows + noCols)) / (noRows * noCols)
    mask_weights[mask_weights < prob] = 0
    mask_weights[mask_weights >= prob] = 1
    noParameters = int(np.sum(mask_weights))
    print("Create Sparse Matrix: No parameters, NoRows, NoCols ", noParameters, noRows, noCols)
    return noParameters, mask_weights.astype(np.float32)



# SET model

class SET_MLP_CIFAR10:
    def __init__(
        self,
        strategy: BaseSETStrategy,
        max_epochs: int,
        use_skip: bool = False,
        init_state: dict | None = None,
    ):
        self.strategy = strategy
        self.use_skip = bool(use_skip)

        # hyperparams
        self.epsilon = 20
        self.zeta = 0.3
        self.batch_size = 100
        self.maxepoches = int(max_epochs)
        self.learning_rate = 0.01
        self.num_classes = 10
        self.momentum = 0.9

        # masks
        self.noPar1, self.wm1_buffer = createWeightsMask(self.epsilon, 32 * 32 * 3, 4000)
        self.noPar2, self.wm2_buffer = createWeightsMask(self.epsilon, 4000, 1000)
        self.noPar3, self.wm3_buffer = createWeightsMask(self.epsilon, 1000, 4000)

        if self.use_skip:
            # skip: input -> layer2 (1000)
            self.noParSkip02, self.wmSkip02_buffer = createWeightsMask(self.epsilon, 32 * 32 * 3, 1000)

 
        self.wm1_core_buffer = np.zeros_like(self.wm1_buffer, dtype=np.float32)
        self.wm2_core_buffer = np.zeros_like(self.wm2_buffer, dtype=np.float32)
        self.wm3_core_buffer = np.zeros_like(self.wm3_buffer, dtype=np.float32)
        if self.use_skip:
            self.wmSkip02_core_buffer = np.zeros_like(self.wmSkip02_buffer, dtype=np.float32)


        self.w1 = self.w2 = self.w3 = self.w4 = None
        self.wSkip02 = None
        self.wSRelu1 = self.wSRelu2 = self.wSRelu3 = None

        # build model
        self.create_model()

        # optionally restore a baseline (for fair bench)
        if init_state is not None:
            self.set_state(init_state)

    # Build model (skip vs no-skip)
    def create_model(self):
        if not self.use_skip:
            self.model = Sequential()
            self.model.add(Flatten(input_shape=(32, 32, 3)))
            self.model.add(Dense(4000, name="sparse_1", kernel_constraint=MaskWeights(self.wm1_buffer)))
            self.model.add(SReLU(name="srelu1"))
            self.model.add(Dropout(0.3))
            self.model.add(Dense(1000, name="sparse_2", kernel_constraint=MaskWeights(self.wm2_buffer)))
            self.model.add(SReLU(name="srelu2"))
            self.model.add(Dropout(0.3))
            self.model.add(Dense(4000, name="sparse_3", kernel_constraint=MaskWeights(self.wm3_buffer)))
            self.model.add(SReLU(name="srelu3"))
            self.model.add(Dropout(0.3))
            self.model.add(Dense(self.num_classes, name="dense_4"))
            self.model.add(Activation("softmax"))
            self._restore_previous_weights()
            return

        # Functional skip model:
        # x -> h1 -> h2_main
        # x -> h2_skip
        # add -> h3 -> out
        inp = Input(shape=(32, 32, 3))
        x = Flatten(name="flatten")(inp)

        h1 = Dense(4000, name="sparse_1", kernel_constraint=MaskWeights(self.wm1_buffer))(x)
        h1 = SReLU(name="srelu1")(h1)
        h1 = Dropout(0.3)(h1)

        h2_main = Dense(1000, name="sparse_2", kernel_constraint=MaskWeights(self.wm2_buffer))(h1)
        h2_skip = Dense(1000, name="skip_02", kernel_constraint=MaskWeights(self.wmSkip02_buffer))(x)
        h2 = Add(name="add_02")([h2_main, h2_skip])
        h2 = SReLU(name="srelu2")(h2)
        h2 = Dropout(0.3)(h2)

        h3 = Dense(4000, name="sparse_3", kernel_constraint=MaskWeights(self.wm3_buffer))(h2)
        h3 = SReLU(name="srelu3")(h3)
        h3 = Dropout(0.3)(h3)

        out = Dense(self.num_classes, name="dense_4")(h3)
        out = Activation("softmax")(out)

        self.model = Model(inputs=inp, outputs=out)
        self._restore_previous_weights()

    def _restore_previous_weights(self):
        names = {l.name for l in self.model.layers}

        def maybe_set(name, weights):
            if weights is not None and name in names:
                self.model.get_layer(name).set_weights(weights)

        maybe_set("sparse_1", self.w1)
        maybe_set("srelu1", self.wSRelu1)
        maybe_set("sparse_2", self.w2)
        maybe_set("srelu2", self.wSRelu2)
        maybe_set("sparse_3", self.w3)
        maybe_set("srelu3", self.wSRelu3)
        maybe_set("dense_4", self.w4)
        if self.use_skip:
            maybe_set("skip_02", self.wSkip02)

            
    def _capture_weights(self):
        self.w1 = self.model.get_layer("sparse_1").get_weights()
        self.w2 = self.model.get_layer("sparse_2").get_weights()
        self.w3 = self.model.get_layer("sparse_3").get_weights()
        self.w4 = self.model.get_layer("dense_4").get_weights()

        self.wSRelu1 = self.model.get_layer("srelu1").get_weights()
        self.wSRelu2 = self.model.get_layer("srelu2").get_weights()
        self.wSRelu3 = self.model.get_layer("srelu3").get_weights()

        if self.use_skip:
            self.wSkip02 = self.model.get_layer("skip_02").get_weights()

    def get_state(self) -> dict:
        # Ensure weights holders are current
        self._capture_weights()

        state = {
            "use_skip": self.use_skip,
            "wm1": self.wm1_buffer.copy(),
            "wm2": self.wm2_buffer.copy(),
            "wm3": self.wm3_buffer.copy(),
            "w1": [a.copy() for a in self.w1],
            "w2": [a.copy() for a in self.w2],
            "w3": [a.copy() for a in self.w3],
            "w4": [a.copy() for a in self.w4],
            "wSRelu1": [a.copy() for a in self.wSRelu1],
            "wSRelu2": [a.copy() for a in self.wSRelu2],
            "wSRelu3": [a.copy() for a in self.wSRelu3],
        }
        if self.use_skip:
            state["wmSkip02"] = self.wmSkip02_buffer.copy()
            state["wSkip02"] = [a.copy() for a in self.wSkip02]
        return state

    def set_state(self, state: dict) -> None:
        if bool(state.get("use_skip", False)) != self.use_skip:
            raise ValueError("State architecture (skip/no-skip) does not match this model.")

        np.copyto(self.wm1_buffer, state["wm1"])
        np.copyto(self.wm2_buffer, state["wm2"])
        np.copyto(self.wm3_buffer, state["wm3"])
        if self.use_skip:
            np.copyto(self.wmSkip02_buffer, state["wmSkip02"])

        # push masks into constraints
        self.model.get_layer("sparse_1").kernel_constraint.update(self.wm1_buffer)
        self.model.get_layer("sparse_2").kernel_constraint.update(self.wm2_buffer)
        self.model.get_layer("sparse_3").kernel_constraint.update(self.wm3_buffer)
        if self.use_skip:
            self.model.get_layer("skip_02").kernel_constraint.update(self.wmSkip02_buffer)

        # restore weights
        self.w1 = [a.copy() for a in state["w1"]]
        self.w2 = [a.copy() for a in state["w2"]]
        self.w3 = [a.copy() for a in state["w3"]]
        self.w4 = [a.copy() for a in state["w4"]]
        self.wSRelu1 = [a.copy() for a in state["wSRelu1"]]
        self.wSRelu2 = [a.copy() for a in state["wSRelu2"]]
        self.wSRelu3 = [a.copy() for a in state["wSRelu3"]]
        if self.use_skip:
            self.wSkip02 = [a.copy() for a in state["wSkip02"]]

        self._restore_previous_weights()

        
    def compute_kernel_grads(self, x_batch, y_batch):
        loss_fn = tf.keras.losses.CategoricalCrossentropy()

        k1 = self.model.get_layer("sparse_1").kernel
        k2 = self.model.get_layer("sparse_2").kernel
        k3 = self.model.get_layer("sparse_3").kernel
        kskip = self.model.get_layer("skip_02").kernel if self.use_skip else None

        with tf.GradientTape() as tape:
            y_pred = self.model(x_batch, training=True)
            loss = loss_fn(y_batch, y_pred)

        if self.use_skip:
            g1, g2, gskip, g3 = tape.gradient(loss, [k1, k2, kskip, k3])
            return [g1.numpy(), g2.numpy(), gskip.numpy(), g3.numpy()]
        else:
            g1, g2, g3 = tape.gradient(loss, [k1, k2, k3])
            return [g1.numpy(), g2.numpy(), g3.numpy()]


    def rewireMask(self, weights_2d, noWeights: int, mask_2d, core_buffer_2d, extra_info: dict):
        # prune modifies mask_2d in-place
        self.strategy.prune_neurons(mask_2d, weights_2d.ravel(), mask_2d.ravel(), extra_info)

        np.copyto(core_buffer_2d, mask_2d)

        noRewires = int(noWeights - np.sum(mask_2d))
        if noRewires <= 0:
            return

        # regrow modifies mask_2d in-place
        self.strategy.regrow_neurons(noRewires, weights_2d.shape, mask_2d, extra_info)


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

        self.rewireMask(self.w1[0], self.noPar1, self.wm1_buffer, self.wm1_core_buffer, ex1)

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

        self.rewireMask(self.w2[0], self.noPar2, self.wm2_buffer, self.wm2_core_buffer, ex2)

        # --- Layer 3 ---
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

        self.rewireMask(self.w3[0], self.noPar3, self.wm3_buffer, self.wm3_core_buffer, ex3)

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
        self.w1[0] *= self.wm1_core_buffer
        self.w2[0] *= self.wm2_core_buffer
        self.w3[0] *= self.wm3_core_buffer
        if self.use_skip:
            self.wSkip02[0] *= self.wmSkip02_core_buffer

        # push masks into constraints
        self.model.get_layer("sparse_1").kernel_constraint.update(self.wm1_buffer)
        self.model.get_layer("sparse_2").kernel_constraint.update(self.wm2_buffer)
        self.model.get_layer("sparse_3").kernel_constraint.update(self.wm3_buffer)
        if self.use_skip:
            self.model.get_layer("skip_02").kernel_constraint.update(self.wmSkip02_buffer)

        # restore weights into model
        self._restore_previous_weights()


    def train(self, target_accuracy: float = 1.0):
        x_train, x_test, y_train, y_test = self.read_data()

        # keep for EMA strategies that read data
        self._ema_x_train = x_train

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
        train_ds = train_ds.map(
            lambda x, y: (data_augmentation(x, training=True), y),
            num_parallel_calls=AUTOTUNE,
        )
        train_ds = train_ds.prefetch(AUTOTUNE)

        val_ds = tf.data.Dataset.from_tensor_slices((x_test, y_test))
        val_ds = val_ds.batch(self.batch_size).prefetch(AUTOTUNE)

        self.accuracies_per_epoch = []
        epoch_count = -1
        best_accuracy = 0.0

        sgd = optimizers.SGD(learning_rate=self.learning_rate, momentum=self.momentum)
        self.model.compile(loss="categorical_crossentropy", optimizer=sgd, metrics=["accuracy"])

        needs_grad = bool(getattr(self.strategy, "needs_gradients", False))

        # init fisher V buffers if needed
        if needs_grad:
            self.strategy.v1 = np.zeros_like(self.wm1_buffer, dtype=np.float32)
            self.strategy.v2 = np.zeros_like(self.wm2_buffer, dtype=np.float32)
            self.strategy.v3 = np.zeros_like(self.wm3_buffer, dtype=np.float32)
            if self.use_skip:
                self.strategy.vSkip02 = np.zeros_like(self.wmSkip02_buffer, dtype=np.float32)

        for epoch in range(0, self.maxepoches):
            hist = self.model.fit(
                train_ds,
                steps_per_epoch=steps_per_epoch,
                epochs=epoch,
                validation_data=val_ds,
                initial_epoch=epoch - 1,
                verbose=1,
            )

            acc = float(hist.history["val_accuracy"][0])
            best_accuracy = max(best_accuracy, acc)
            self.accuracies_per_epoch.append(acc)

            if acc >= target_accuracy:
                epoch_count = epoch
                break

            fisher_payload = None
            if needs_grad:
                bs = 256
                grads = self.compute_kernel_grads(x_train[:bs], y_train[:bs])
                b = float(getattr(self.strategy, "beta", 0.9))

                if self.use_skip:
                    g1, g2, gskip, g3 = grads
                    self.strategy.v1 = b * self.strategy.v1 + (1.0 - b) * (g1.astype(np.float32) ** 2)
                    self.strategy.v2 = b * self.strategy.v2 + (1.0 - b) * (g2.astype(np.float32) ** 2)
                    self.strategy.vSkip02 = b * self.strategy.vSkip02 + (1.0 - b) * (gskip.astype(np.float32) ** 2)
                    self.strategy.v3 = b * self.strategy.v3 + (1.0 - b) * (g3.astype(np.float32) ** 2)

                    fisher_payload = {
                        "grads": (g1, g2, gskip, g3),
                        "Vs": (self.strategy.v1, self.strategy.v2, self.strategy.vSkip02, self.strategy.v3),
                    }
                else:
                    g1, g2, g3 = grads
                    self.strategy.v1 = b * self.strategy.v1 + (1.0 - b) * (g1.astype(np.float32) ** 2)
                    self.strategy.v2 = b * self.strategy.v2 + (1.0 - b) * (g2.astype(np.float32) ** 2)
                    self.strategy.v3 = b * self.strategy.v3 + (1.0 - b) * (g3.astype(np.float32) ** 2)

                    fisher_payload = {
                        "grads": (g1, g2, g3),
                        "Vs": (self.strategy.v1, self.strategy.v2, self.strategy.v3),
                    }

            self.weightsEvolution(fisher_payload=fisher_payload)

        self.accuracies_per_epoch = np.asarray(self.accuracies_per_epoch, dtype=np.float32)
        return epoch_count, best_accuracy

    def read_data(self):
        (x_train, y_train), (x_test, y_test) = cifar10.load_data()
        y_train = to_categorical(y_train, self.num_classes)
        y_test = to_categorical(y_test, self.num_classes)
        x_train = x_train.astype("float32")
        x_test = x_test.astype("float32")

        xTrainMean = np.mean(x_train, axis=0)
        xTrainStd = np.std(x_train, axis=0)
        x_train = (x_train - xTrainMean) / xTrainStd
        x_test = (x_test - xTrainMean) / xTrainStd

        return x_train, x_test, y_train, y_test
