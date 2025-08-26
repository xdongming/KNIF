import numpy as np
import tensorflow as tf
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.layers import Layer, Dense, Input
from tensorflow.keras.models import Model
tf.keras.backend.set_floatx('float64')

class KoopmanGeneralSolver(object):
    '''
    Build the Koopman solver
    '''

    def __init__(self, dic, target_dim, reg=0.0):
        self.dic = dic                 
        self.dic_func = dic.call
        self.target_dim = target_dim  
        self.reg = reg

    def separate_data(self, data):
        data_x = data[0]
        data_y = data[1]
        return data_x, data_y

    def build(self, data_train):
        self.data_train = data_train
        self.data_x_train, self.data_y_train = self.separate_data(self.data_train)
        self.compute_final_info(reg_final=0.0)

    def compute_final_info(self, reg_final):
        self.K = self.compute_K(self.dic_func, self.data_x_train, self.data_y_train, reg=reg_final)
        self.eig_decomp(self.K)
        self.compute_mode()

    def eig_decomp(self, K):
        self.eigenvalues, self.eigenvectors = np.linalg.eig(K)
        idx = self.eigenvalues.real.argsort()[::-1]
        self.eigenvalues = self.eigenvalues[idx]
        self.eigenvectors = self.eigenvectors[:, idx]
        self.eigenvectors_inv = np.linalg.inv(self.eigenvectors)

    def eigenfunctions(self, data_x):
        psi_x = self.dic_func(data_x)               
        if len(psi_x.shape) == 3:
            psi_x = tf.squeeze(psi_x, axis=1)        
        psi_x = np.array(psi_x)                  
        val = np.matmul(psi_x, self.eigenvectors)   
        return val

    def compute_mode(self):
        self.B = self.dic.generate_B(self.data_x_train)
        self.modes = np.matmul(self.eigenvectors_inv, self.B).T   # [target_dim, D]
        return self.modes

    def calc_psi_next(self, data_x, K):
        psi_x = self.dic_func(data_x)
        if len(psi_x.shape) == 3:
            shape = tf.shape(psi_x)
            B = shape[0]
            T = shape[1]
            D = shape[2]
            psi_x = tf.reshape(psi_x, [B*T, D])
        psi_next = tf.matmul(psi_x, K)
        return psi_next

    def predict(self, x0, traj_len):
        traj = [x0]
        for _ in range(traj_len - 1):
            x_curr = traj[-1]                             # [B, target_dim]
            efunc = self.eigenfunctions(x_curr)          # [B, D]
            x_next = np.matmul(self.modes, (self.eigenvalues * efunc).T)  # [target_dim, B]
            traj.append((x_next.real).T)                  # [B, target_dim]
        traj = np.transpose(np.stack(traj, axis=0), [1, 0, 2])  # [B, T, target_dim]
        return traj.squeeze()

    def compute_K(self, dic, data_x, data_y, reg):
        psi_x = dic(data_x)   
        psi_y = dic(data_y)

        # 3D -> 2D (B*T', D)
        if len(psi_x.shape) == 3:
            shape = tf.shape(psi_x)
            B = shape[0]
            T = shape[1]
            D = shape[2]
            psi_x = tf.reshape(psi_x, [B*T, D])
            psi_y = tf.reshape(psi_y, [B*T, D])

        psi_xt = tf.transpose(psi_x)                           # [D,N]
        idmat = tf.eye(tf.shape(psi_x)[1], dtype='float64')    # [N,N]
        xtx_inv = tf.linalg.pinv(reg * idmat + tf.matmul(psi_xt, psi_x))
        xty = tf.matmul(psi_xt, psi_y)
        self.K_reg = tf.matmul(xtx_inv, xty)                   # [D,D]
        return self.K_reg


class KoopmanDLSolver(KoopmanGeneralSolver):
    '''
    Build the Koopman model with dictionary learning
    '''
    def build_model(self):
        inputs_x = Input((None, self.target_dim))   # [B,T,V]
        inputs_y = Input((None, self.target_dim))   # [B,T,V]

        class PsiNNLayer(Layer):
            def __init__(self, psinn):
                super().__init__()
                self.psinn = psinn
            def call(self, inputs):
                return self.psinn.call(inputs)      # [B,T',D]

        psi_layer = PsiNNLayer(self.dic)

        psi_x = psi_layer(inputs_x)                 # [B,T',D]
        psi_y = psi_layer(inputs_y)                 # [B,T',D]

        Layer_K = Dense(units=psi_y.shape[-1], use_bias=False, name='Layer_K', trainable=False)
        psi_next = Layer_K(psi_x)                   # [B,T',D]

        outputs = psi_next - psi_y                  # [B,T',D]
        model = Model(inputs=[inputs_x, inputs_y], outputs=outputs)
        return model

    def train_psi(self, model, epochs):
        history = model.fit(
            x=self.data_train,
            y=self.zeros_data_y_train,
            epochs=epochs,
            validation_data=(self.data_valid, self.zeros_data_y_valid),
            batch_size=self.batch_size,
            verbose=1
        )
        return history

    def build(self, data_train, data_valid, epochs, batch_size, lr, log_interval, lr_decay_factor):
        self.data_train = data_train
        self.data_x_train, self.data_y_train = self.separate_data(self.data_train)

        self.data_valid = data_valid
        self.zeros_data_y_train = tf.zeros_like(self.dic_func(self.data_y_train))
        self.zeros_data_y_valid = tf.zeros_like(self.dic_func(self.data_valid[1]))
        self.batch_size = batch_size

        self.model = self.build_model()
        opt = Adam(lr)
        self.model.compile(optimizer=opt, loss='mse')

        losses = []
        for i in range(epochs):
            self.K = self.compute_K(self.dic_func, self.data_x_train, self.data_y_train, self.reg)
            self.model.get_layer('Layer_K').weights[0].assign(self.K)

            self.history = self.train_psi(self.model, epochs=2)

            print('number of the outer loop:', i)
            if i % log_interval == 0:
                losses.append(self.history.history['loss'][-1])

                if len(losses) > 2 and losses[-1] > losses[-2]:
                    print("Error increased. Decay learning rate")
                    curr_lr = self.model.optimizer.learning_rate * lr_decay_factor
                    self.model.optimizer.learning_rate.assign(curr_lr)

        self.compute_final_info(reg_final=0.01)
