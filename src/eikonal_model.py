import tensorflow.compat.v1 as tf
import numpy as np
import os

tf.disable_v2_behavior()
os.environ["CUDA_VISIBLE_DEVICES"] = "-1"

tf.set_random_seed(1234)
np.random.seed(1234)

class Eikonal2DnetCV2:
    def __init__(self, x, y, x_e, y_e, T_e, layers, model_type="physics", C=1.0, alpha=1e-5, alphaL2=1e-6, jobs=4):
        """Initialize the Eikonal2DnetCV2 model.
        
        Args:
            x, y: Input coordinates for the domain.
            x_e, y_e, T_e: Sparse measurement points and true activation times.
            layers: List defining the neural network architecture.
            model_type: "data" for Model 1, "physics" for Model 2 (default).
            C, alpha, alphaL2, jobs: Hyperparameters for scaling and optimization.
        """
        X = np.concatenate([x, y], 1)
        self.lb = X.min(0)
        self.ub = X.max(0)
        
        self.X = X
        self.x = x
        self.y = y
        self.T_e = T_e
        self.x_e = x_e
        self.y_e = y_e
        self.layers = layers
        self.model_type = model_type
        
        self.weights, self.biases = self.initialize_NN(layers)
        
        self.C = tf.constant(C)
        self.alpha = tf.constant(alpha)
        self.alphaL2 = alphaL2
        
        self.sess = tf.Session(config=tf.ConfigProto(allow_soft_placement=True,
                                                    intra_op_parallelism_threads=jobs,
                                                    inter_op_parallelism_threads=jobs,
                                                    device_count={'CPU': jobs}))
        
        self.x_tf = tf.placeholder(tf.float32, shape=[None, self.x.shape[1]])
        self.y_tf = tf.placeholder(tf.float32, shape=[None, self.y.shape[1]])
        self.T_e_tf = tf.placeholder(tf.float32, shape=[None, self.T_e.shape[1]]) 
        self.x_e_tf = tf.placeholder(tf.float32, shape=[None, self.x_e.shape[1]]) 
        self.y_e_tf = tf.placeholder(tf.float32, shape=[None, self.y_e.shape[1]]) 
        
        self.T_pred, self.f_T_pred = self.net_eikonal(self.x_tf, self.y_tf)
        self.T_e_pred, self.f_T_e_pred = self.net_eikonal(self.x_e_tf, self.y_e_tf)
        
        if model_type == "data":
            self.loss = tf.reduce_mean(tf.square(self.T_e_tf - self.T_e_pred))
        else:  # physics
            self.loss = tf.reduce_mean(tf.square(self.T_e_tf - self.T_e_pred)) + \
                       tf.reduce_mean(tf.square(self.f_T_e_pred)) + \
                       tf.reduce_mean(tf.square(self.f_T_pred)) + \
                       sum([self.alphaL2*tf.nn.l2_loss(w) for w in self.weights])
        
        self.optimizer_Adam = tf.train.AdamOptimizer()
        self.train_op_Adam = self.optimizer_Adam.minimize(self.loss)
        
        init = tf.global_variables_initializer()
        self.sess.run(init)
        self.lossit = []

    def initialize_NN(self, layers):      
        def xavier_init(size):
            in_dim = size[0]
            out_dim = size[1]
            xavier_stddev = 1. / np.sqrt((in_dim + out_dim) / 2.)
            return tf.Variable(tf.random_normal([in_dim, out_dim], dtype=tf.float32) * xavier_stddev, dtype=tf.float32)   
        
        weights = []
        biases = []
        num_layers = len(layers) 
        for l in range(0, num_layers-1):
            W = xavier_init(size=[layers[l], layers[l+1]])
            b = tf.Variable(tf.zeros([1, layers[l+1]], dtype=tf.float32), dtype=tf.float32)
            weights.append(W)
            biases.append(b)        
        return weights, biases
    
    def neural_net(self, X, weights, biases):
        num_layers = len(weights) + 1
        
        H = 2.0 * (X - self.lb) / (self.ub - self.lb) - 1.0
        for l in range(0, num_layers-2):
            W = weights[l]
            b = biases[l]
            H = tf.tanh(tf.add(tf.matmul(H, W), b))
        W = weights[-1]
        b = biases[-1]
        Y = tf.add(tf.matmul(H, W), b)
        return Y
    
    def net_eikonal(self, x, y):
        T = self.neural_net(tf.concat([x, y], 1), self.weights, self.biases)
        
        T_x = tf.gradients(T, x)[0]
        T_y = tf.gradients(T, y)[0]
        
        f_T = tf.sqrt(T_x**2 + T_y**2) - 1.0
        
        return T, f_T
    
    def callback(self, loss):
        self.lossit.append(loss)
        print('Loss: %.5e' % (loss))
    
    def train(self):
        tf_dict = {self.x_tf: self.x, self.y_tf: self.y, 
                   self.x_e_tf: self.x_e, self.y_e_tf: self.y_e, self.T_e_tf: self.T_e}
        
        def loss_and_grad(w):
            feed_dict = {self.x_tf: self.x, self.y_tf: self.y, 
                         self.x_e_tf: self.x_e, self.y_e_tf: self.y_e, self.T_e_tf: self.T_e}
            loss_val, grad_val = self.sess.run([self.loss] + tf.gradients(self.loss, self.weights), feed_dict=feed_dict)
            return loss_val, grad_val
        
        initial_vars = self.sess.run(self.weights)
        initial_params = np.hstack([v.flatten() for v in initial_vars])
        
        result = minimize(lambda w: loss_and_grad(w)[0], initial_params,
                          jac=lambda w: np.hstack(loss_and_grad(w)[1]),
                          method='L-BFGS-B',
                          options={'maxiter': 10000, 'maxfun': 50000, 'maxcor': 50, 'maxls': 50, 'ftol': 1.0 * np.finfo(float).eps},
                          callback=lambda x: self.callback(loss_and_grad(x)[0]))
        
        start_idx = 0
        for var, w in zip(self.weights, result.x):
            flat_size = np.prod(var.shape.as_list())
            var_value = w[start_idx:start_idx + flat_size].reshape(var.shape)
            start_idx += flat_size
            self.sess.run(tf.assign(var, var_value))
    
    def train_Adam(self, nIter): 
        self.lossit = []

        tf_dict = {self.x_tf: self.x, self.y_tf: self.y, 
                   self.x_e_tf: self.x_e, self.y_e_tf: self.y_e, self.T_e_tf: self.T_e}        
        start_time = time.time()
        for it in range(nIter):
            self.sess.run(self.train_op_Adam, tf_dict)
            loss_value = self.sess.run(self.loss, tf_dict)
            self.lossit.append(loss_value)

            if it % 10 == 0:
                elapsed = time.time() - start_time
                print('It: %d, Loss: %.3e, Time: %.2f' % 
                      (it, loss_value, elapsed))
                start_time = time.time()
    
    def predict(self, x_star, y_star):
        tf_dict = {self.x_tf: x_star, self.y_tf: y_star,
                   self.x_e_tf: self.x_e, self.y_e_tf: self.y_e}
        
        T_star = self.sess.run(self.T_pred, tf_dict)

        return T_star
    
    def save_results(self, filepath="eikonal_results.pkl"):
        weights = self.sess.run(self.weights)
        biases = self.sess.run(self.biases)
        data = {
            "weights": weights,
            "biases": biases,
            "loss_history": self.lossit,
            "x": self.x,
            "y": self.y,
            "x_e": self.x_e,
            "y_e": self.y_e,
            "T_e": self.T_e
        }
        with open(filepath, 'wb') as f:
            pickle.dump(data, f)
        print(f"Results saved to {filepath}")
