from scipy.cluster.vq import kmeans
import scipy
from tensorflow.keras.layers import Layer, Dense, Conv1D
from tensorflow.keras.layers import Input, Add, Multiply, Lambda, Concatenate
from tensorflow.keras.models import Model
import numpy as np
import tensorflow as tf
tf.keras.backend.set_floatx('float64')


class AbstractDictionary(object):
    def generate_B(self, inputs):
        target_dim = inputs.shape[-1]
        self.basis_func_number = self.n_dic_customized + target_dim + 1
        # Form B matrix
        self.B = np.zeros((self.basis_func_number, target_dim))
        for i in range(0, target_dim):
            self.B[i + 1][i] = 1
        return self.B


class DicNN(Layer):
    """Trainable dictionaries

    """

    def __init__(self, layer_sizes=[64, 64], n_psi_train=22, **kwargs):
        super(DicNN, self).__init__(**kwargs)
        self.layer_sizes = layer_sizes
        self.input_layer = Dense(self.layer_sizes[0], use_bias=False)
        self.hidden_layers = [Dense(s, activation='tanh') for s in layer_sizes]
        self.output_layer = Dense(n_psi_train)
        self.n_psi_train = n_psi_train

    def call(self, inputs):
        psi_x_train = self.input_layer(inputs)
        for layer in self.hidden_layers:
            psi_x_train = psi_x_train + layer(psi_x_train)
        outputs = self.output_layer(psi_x_train)
        return outputs

    def get_config(self):
        config = super(DicNN, self).get_config()
        config.update({
            'layer_sizes': self.layer_sizes,
            'n_psi_train': self.n_psi_train
        })
        return config
    
class PsiNNWithEncoder(Layer):
    def __init__(self, visible_dim, num_hidden=1, hidden_size=32, kernel_size=31,
                 layer_sizes=[64, 64], n_psi_train=22, target_dim=None, **kwargs):
        super().__init__(**kwargs)
        self.visible_dim = visible_dim
        self.num_hidden = num_hidden
        self.hidden_size = hidden_size
        self.kernel_size = kernel_size
        self.layer_sizes = layer_sizes
        self.n_dic_customized = n_psi_train
        self.target_dim = target_dim if target_dim is not None else visible_dim

        self.conv1 = Conv1D(filters=self.hidden_size, kernel_size=self.kernel_size,
                            activation="relu", padding="valid")
        self.conv2 = Conv1D(filters=self.hidden_size, kernel_size=1,
                            activation="relu", padding="valid")
        self.conv3 = Conv1D(filters=self.num_hidden, kernel_size=1,
                            activation="tanh", padding="valid")

        self.dicNN = DicNN(layer_sizes=self.layer_sizes, n_psi_train=self.n_dic_customized)

    def _tile_for_prediction(self, inputs_2d):
        x = tf.expand_dims(inputs_2d, axis=1)           # [B,1,V]
        x = tf.tile(x, [1, self.kernel_size, 1])        # [B,K,V]
        return x

    def call(self, inputs):
        vis = self._tile_for_prediction(inputs) if len(inputs.shape) == 2 else inputs[..., :self.visible_dim]

        hid = self.conv1(vis)
        hid = self.conv2(hid)
        hid = self.conv3(hid)                           # [B,T',H]

        interval = (self.kernel_size - 1) // 2
        vis_cropped = (vis[:, interval:-interval, :]
                       if (vis.shape[1] is None or int(vis.shape[1]) > self.kernel_size)
                       else vis[:, interval:interval+1, :])  # [B,T',V]

        aug_inputs = Concatenate(axis=-1)([vis_cropped, hid])  # [B,T', V+H]

        const = tf.ones((tf.shape(aug_inputs)[0], tf.shape(aug_inputs)[1], 1), dtype=aug_inputs.dtype)
        psi_x = self.dicNN(aug_inputs)                        
        outputs = Concatenate(axis=-1)([const, aug_inputs, psi_x])
        return outputs

    def generate_B(self, inputs):
        aug_dim = self.visible_dim + self.num_hidden
        self.basis_func_number = self.n_dic_customized + aug_dim + 1
        B = np.zeros((self.basis_func_number, self.target_dim))
        for i in range(self.target_dim):
            B[i + 1][i] = 1.0
        return B

class PsiNN(Layer, AbstractDictionary):
    """Concatenate constant, data and trainable dictionaries together as [1, data, DicNN]

    """

    def __init__(
            self,
            dic_trainable=DicNN,
            layer_sizes=[
                64,
                64],
            n_psi_train=22,
            **kwargs):
        super(PsiNN, self).__init__(**kwargs)
        self.layer_sizes = layer_sizes
        self.dic_trainable = dic_trainable
        self.n_dic_customized = n_psi_train
        self.dicNN = self.dic_trainable(
            layer_sizes=self.layer_sizes,
            n_psi_train=self.n_dic_customized)

    def call(self, inputs):
        constant = tf.ones((tf.shape(inputs)[0], 1), dtype=inputs.dtype)
        psi_x_train = self.dicNN(inputs)
        outputs = Concatenate()([constant, inputs, psi_x_train])
        return outputs

    def get_config(self):
        config = super(PsiNN, self).get_config()
        config.update({
            'dic_trainable': self.dic_trainable,
            'layer_sizes': self.layer_sizes,
            'n_psi_train': self.n_dic_customized
        })
        return config


class DicRBF(AbstractDictionary):
    """
    RBF based on notations in
    (https://en.wikipedia.org/wiki/Radial_basis_function)
    """

    def __init__(self, rbf_number=100, regularizer=1e-4):
        self.n_dic_customized = rbf_number
        self.regularizer = regularizer

    def build(self, data):
        self.centers, residual = kmeans(data, self.n_dic_customized)

    def call(self, data):
        rbfs = []
        for n in range(self.centers.shape[0]):
            r = scipy.spatial.distance.cdist(
                data, np.matrix(self.centers[n, :]))
            rbf = scipy.special.xlogy(r**2, r + self.regularizer)
            rbfs.append(rbf)

        rbfs = tf.transpose(tf.squeeze(rbfs))
        rbfs = tf.reshape(rbfs, shape=(data.shape[0], -1))

        ones = tf.ones(shape=(rbfs.shape[0], 1), dtype='float64')
        results = tf.concat([ones, data, rbfs], axis=-1)
        return results
