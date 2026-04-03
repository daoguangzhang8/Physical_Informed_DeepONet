# import sys
# from PIL import Image
# import tensorflow_probability as tfp
# import matplotlib as mpl
# mpl.use('Agg')
# import scipy.special as ss
# from scipy.interpolate import griddata
# from scipy import sparse
# import scipy.signal as ssignal
# from itertools import product, combinations
# from mpl_toolkits.mplot3d import Axes3D
# from mpl_toolkits.mplot3d.art3d import Poly3DCollection
# from mpl_toolkits.axes_grid1 import make_axes_locatable
# import matplotlib.gridspec as gridspec
import math
import tensorflow.compat.v1 as tf
import tensorflow as tf2
import numpy as np
import matplotlib.pyplot as plt
import scipy.io
import time

tf.disable_v2_behavior()


class PhysicsInformedNN:
    # Initialize the class
    def __init__(self, x, y, layers, v, scu0r, scu0i):

        X = np.concatenate([x, y], 1)  # inputs for training

        self.lb = X.min(0)
        self.upb = X.max(0)

        self.X = X
        self.x = X[:, 0:1]
        self.y = X[:, 1:2]
        self.layers = layers
        self.v = v
        self.u0r = scu0r
        self.u0i = scu0i

        # tf placeholders and graph
        # config=tf.ConfigProto()
        # config.gpu_options.allow_growth = True
        # config.gpu_options.per_process_gpu_memory_fraction = 0.8
        # self.sess = tf.InteractiveSession(config=config)
        self.sess = tf.Session(config=tf.ConfigProto(allow_soft_placement=True,
                                                     log_device_placement=True))
        self.x_tf = tf.placeholder(tf.float64, shape=[None, self.x.shape[1]])
        self.y_tf = tf.placeholder(tf.float64, shape=[None, self.y.shape[1]])
        self.xb_tf = tf.placeholder(tf.float64, shape=[None, self.x.shape[1]])
        self.yb_tf = tf.placeholder(tf.float64, shape=[None, self.y.shape[1]])
        self.v_tf = tf.placeholder(tf.float64, shape=[None, self.v.shape[1]])
        self.u0r_tf = tf.placeholder(tf.float64, shape=[None, self.u0r.shape[1]])
        self.u0i_tf = tf.placeholder(tf.float64, shape=[None, self.u0i.shape[1]])

        Q = 15
        alpha = 1 / Q
        # alpha = 0
        self.f = 5
        fr = 50
        rhot = (1 - alpha / np.pi * np.log(self.f / fr) - 1j * alpha / 2) ** 2
        self.m0 = 1 / 1.5 ** 2
        self.m0r = self.m0 * np.real(rhot)
        self.m0i = self.m0 * np.imag(rhot)
        self.pn = 1  # number of quadratic layers

        ########################################################
        ################# different cases ######################
        ########################################################
        # smooth velocity
        self.m = 1 / self.v_tf ** 2  # non-smooth
        self.mr = self.m * np.real(rhot)
        self.mi = self.m * np.imag(rhot)

        # Initialize linear NN, output and loss function
        # self.weights, self.biases, self.wb = self.initialize_NN(layers)  # initialize linear network
        # self.urnet, self.uinet, self.fu1, self.fu2 = self.net_NS(self.x_tf, self.y_tf)   #  linear
        # self.urnet, self.uinet, self.fu1, self.fu2 = self.net_NSpml(self.x_tf, self.y_tf)  #  linear, pml
        # self.urnet, self.uinet, self.fu1, self.fu2 = self.net_NSmri(self.x_tf, self.y_tf)   # linear, complex velocity
        # self.urnet, self.uinet, self.fu1, self.fu2 = self.net_NSmripml(self.x_tf,self.y_tf)  # linear, complex velocity, pml

        # Initialize quadratic NN, output and loss function
        self.weights, self.biases, self.wb = self.initialize_NNq(layers)  # initialize quadratic network
        # self.urnet, self.uinet, self.fu1, self.fu2 = self.net_NSq(self.x_tf, self.y_tf)    # quadratic
        # self.urnet, self.uinet, self.fu1, self.fu2 = self.net_NSpmlq(self.x_tf,self.y_tf)  #  quadratic, pml
        # self.urnet, self.uinet, self.fu1, self.fu2 = self.net_NSmriq(self.x_tf, self.y_tf)   # quadratic, complex velocity
        self.urnet, self.uinet, self.fu1, self.fu2 = self.net_NSmripmlq(self.x_tf, self.y_tf)   # quadratic, complex velocity, pml

        # boundary conditions (free surface)
        # self.urb, self.uib = self.Freesurface(self.xb_tf,self.yb_tf)

        #########################################################
        #########################################################

        # loss function
        self.loss = tf.reduce_sum(tf.square(self.fu1)) + tf.reduce_sum(tf.square(self.fu2))# + \

                    #0.001*(tf.reduce_sum(tf.square(self.urb)) + tf.reduce_sum(tf.square(self.uib)))

        # add Adam
        self.optimizer_Adam = tf.train.AdamOptimizer()
        self.train_op_Adam = self.optimizer_Adam.minimize(self.loss)

        # initial tf variables
        init = tf.global_variables_initializer()
        self.sess.run(init)

    # load parameters from a trained network
    def initialize_Net(self):
        weights = []
        biases = []
        for fileit in range(11):
            bnp = np.load('./test2/badam' + str(fileit) + '.npy')
            wnp = np.load('./test2/wadam' + str(fileit) + '.npy')
            W = tf.Variable(wnp, dtype=tf.float64)
            tf.add_to_collection(tf.GraphKey.WEIGHTS, 'W' + str(fileit))
            b = tf.Variable(bnp, dtype=tf.float64)
            weights.append(W)
            biases.append(b)
        return weights, biases

    # initial the parameters of quadratic network before training
    def initialize_NNq(self, layers):  # give the initial value of net parameters
        weights = []
        biases = []
        num_layers = len(layers)
        # quadratic layer
        for l in range(0, self.pn):
            W = self.xavier_unif_init(size=[layers[l] * (layers[l] + 3) / 2, layers[l + 1]])
            b = tf.Variable(tf.zeros([1, layers[l + 1]], dtype=tf.float64), dtype=tf.float64)
            weights.append(W)
            biases.append(b)
        # traditional liear layer
        for l in range(self.pn, num_layers - 1):
            W = self.xavier_unif_init(size=[layers[l], layers[l + 1]])
            b = tf.Variable(tf.zeros([1, layers[l + 1]], dtype=tf.float64), dtype=tf.float64)
            weights.append(W)
            biases.append(b)
        wb = tf.Variable(tf.ones([1, 2], dtype=tf.float64), dtype=tf.float64)
        return weights, biases, wb

    # initial the parameters of linear network before training
    def initialize_NN(self, layers):
        weights = []
        biases = []
        num_layers = len(layers)
        for l in range(0, num_layers - 1):
            W = self.xavier_unif_init(size=[layers[l], layers[l + 1]])
            b = tf.Variable(tf.zeros([1, layers[l + 1]], dtype=tf.float64), dtype=tf.float64)
            weights.append(W)
            biases.append(b)
        wb = tf.Variable(tf.ones([1, 2], dtype=tf.float64), dtype=tf.float64)
        return weights, biases, wb

    # two ways to give the initial value of net parameters
    def xavier_gaus_init(self, size):
        in_dim = int(size[0])
        out_dim = int(size[1])
        xavier_stddev = np.sqrt(2 / (in_dim + out_dim))
        return tf.Variable(10 * tf.random.truncated_normal([in_dim, out_dim], stddev=xavier_stddev, dtype=tf.float64),
                           dtype=tf.float64)

    def xavier_unif_init(self, size):
        in_dim = int(size[0])
        out_dim = int(size[1])
        xavier_stddev = 1.5 * np.sqrt(6 / (in_dim + out_dim))
        return tf.Variable(
            tf.random.uniform([in_dim, out_dim], minval=-1 * xavier_stddev, maxval=xavier_stddev, dtype=tf.float64),
            dtype=tf.float64)

    # define the operation inside of the quadratic network
    def neural_netqw5(self, x, y, weights, biases):
        num_layers = len(weights)
        Hx = (2 * (x - self.lb[0]) / (self.upb[0] - self.lb[0]) - 1)  # normalization X
        Hy = (2 * (y - self.lb[1]) / (self.upb[1] - self.lb[1]) - 1)
        # quadratic layers
        H = tf.concat([Hx, Hy], 1)
        HC = H
        for l in range(0, self.pn):  # computation in each quadratic layer
            for ih in range(0, H.shape[1]):
                for jh in range(0, H.shape[1] - ih):
                    HC = tf.concat(
                        [HC, tf.expand_dims(H[:, ih], 1) * tf.expand_dims(H[:, ih + jh], 1)], 1)
            W = weights[l]
            b = biases[l]
            HC = tf.tanh(tf.add(tf.matmul(HC, W), b))
        # linear layers
        for l in range(self.pn, num_layers - 1):  # computation in each linear layer
            W = weights[l]
            b = biases[l]
            HC = tf.tanh(tf.add(tf.matmul(HC, W), b))
        W = weights[-1]
        b = biases[-1]
        Y = tf.add(tf.matmul(HC, W), b)
        return Y

    def neural_netqw2(self, x, y, weights, biases):
        num_layers = len(weights)
        Hx = (2 * (x - self.lb[0]) / (self.upb[0] - self.lb[0]) - 1)  # normalization X
        Hy = (2 * (y - self.lb[1]) / (self.upb[1] - self.lb[1]) - 1)
        # quadratic layers
        H = tf.concat([Hx, Hy], 1)
        HC = H
        for l in range(0, self.pn):  # computation in each quadratic layer
            W = weights[l]
            b = biases[l]
            HC = tf.tanh(tf.add(tf.matmul(HC, W), b) ** 2)
        # linear layers
        for l in range(self.pn, num_layers - 1):  # computation in each linear layer
            W = weights[l]
            b = biases[l]
            HC = tf.tanh(tf.add(tf.matmul(HC, W), b))
        W = weights[-1]
        b = biases[-1]
        Y = tf.add(tf.matmul(HC, W), b)
        return Y

    def neural_netqw5qw2all(self, x, y, weights, biases, wb):
        num_layers = len(weights)
        Hx = (2 * (x - self.lb[0]) / (self.upb[0] - self.lb[0]) - 1)  # normalization X
        Hy = (2 * (y - self.lb[1]) / (self.upb[1] - self.lb[1]) - 1)
        # quadratic layers
        H = tf.concat([Hx, Hy], 1)
        HC = H
        for l in range(0, self.pn):  # computation in each quadratic layer
            for ih in range(0, H.shape[1]):
                for jh in range(0, H.shape[1] - ih):
                    HC = tf.concat(
                        [HC, tf.expand_dims(H[:, ih], 1) * tf.expand_dims(H[:, ih + jh], 1)], 1)
            W = weights[l]
            b = biases[l]
            HC = tf.tanh(tf.add(tf.matmul(HC, W), b))
        # linear layers
        for l in range(self.pn, num_layers - 1):  # computation in each quadratic layer
            if (l % 2) == 0:
                W = weights[l]
                b = biases[l]
                HC = tf.tanh(tf.add(tf.matmul(HC, W), b) ** 2)
            else:
                W = weights[l]
                b = biases[l]
                HC = tf.tanh(tf.add(tf.matmul(HC, W), b))
        Fx = (110 * (x - self.lb[0]) / (self.upb[0] - self.lb[0]))
        # wbb = 0.2*tf.ones([1, 2], dtype=tf.float64)
        Fc = tf.tanh(tf.matmul(Fx,wb))
        W = weights[-1]
        b = biases[-1]
        Y = Fc*tf.add(tf.matmul(HC, W), b)
        return Y

    def neural_netqw2all(self, x, y, weights, biases):
        num_layers = len(weights)
        Hx = (2 * (x - self.lb[0]) / (self.upb[0] - self.lb[0]) - 1)  # normalization X
        Hy = (2 * (y - self.lb[1]) / (self.upb[1] - self.lb[1]) - 1)
        # quadratic layers
        H = tf.concat([Hx, Hy], 1)
        HC = H
        for l in range(0, num_layers - 1):  # computation in each quadratic layer
            if (l % 2) == 0:
                W = weights[l]
                b = biases[l]
                HC = tf.tanh(tf.add(tf.matmul(HC, W), b) ** 2)
            else:
                W = weights[l]
                b = biases[l]
                HC = tf.tanh(tf.add(tf.matmul(HC, W), b))
        W = weights[-1]
        b = biases[-1]
        Y = tf.add(tf.matmul(HC, W), b)
        return Y

    # define the operation inside of the linear network
    def neural_net(self, x, y, weights, biases, wb):
        num_layers = len(weights)
        Hx = (2 * (x - self.lb[0]) / (self.upb[0] - self.lb[0]) - 1)  # normalization X
        Hy = (2 * (y - self.lb[1]) / (self.upb[1] - self.lb[1]) - 1)
        H = tf.concat([Hx, Hy], 1)
        HC = H
        for l in range(0, num_layers - 1):  # computation in each layer
            W = weights[l]
            b = biases[l]
            HC = tf.tanh(tf.add(tf.matmul(HC, W), b))
        Fx = (110 * (x - self.lb[0]) / (self.upb[0] - self.lb[0]))
        # wbb = 0.2*tf.ones([1, 2], dtype=tf.float64)
        Fc = tf.tanh(tf.matmul(Fx,wb))
        W = weights[-1]
        b = biases[-1]
        Y = Fc*tf.add(tf.matmul(HC, W), b)
        return Y

    # wave equations
    def net_NS(self, x, y):
        # get the output of network
        uu = self.neural_net(x, y, self.weights, self.biases)  # output of linear network
        uur_pred = uu[:, 0:1]
        uui_pred = uu[:, 1:2]
        # gradients
        ur_x = tf.gradients(uur_pred, x)[0]
        ur_y = tf.gradients(uur_pred, y)[0]
        ur_xx = tf.gradients(ur_x, x)[0]
        ur_yy = tf.gradients(ur_y, y)[0]
        ui_x = tf.gradients(uui_pred, x)[0]
        ui_y = tf.gradients(uui_pred, y)[0]
        ui_xx = tf.gradients(ui_x, x)[0]
        ui_yy = tf.gradients(ui_y, y)[0]
        # scattered wave equations
        f_ur_ac = (ur_xx + ur_yy) + (2 * math.pi * self.f * 1e-3) ** 2 * self.m * uur_pred + (
                2 * math.pi * self.f * 1e-3) ** 2 * (self.m - self.m0) * self.u0r_tf
        f_ui_ac = (ui_xx + ui_yy) + (2 * math.pi * self.f * 1e-3) ** 2 * self.m * uui_pred + (
                2 * math.pi * self.f * 1e-3) ** 2 * (self.m - self.m0) * self.u0i_tf

        return uur_pred, uui_pred, f_ur_ac, f_ui_ac  # ,f_ur_gradx,f_ur_grady,f_ui_gradx,f_ui_grady

    # wave equations with pml
    def net_NSpml(self, x, y):
        self.a0 = 1.79
        self.f0 = 10
        C = self.a0 * self.f0 / self.f

        # get the output of network
        uu = self.neural_net(x, y, self.weights, self.biases)  # linear
        uur_pred = uu[:, 0:1]
        uui_pred = uu[:, 1:2]

        # pml setting
        lx = tf.nn.relu((9.5 * 20 - x) / (9.5 * 20)) + tf.nn.relu((x - 109.5 * 20) / (9.5 * 20))
        ly = tf.nn.relu((9.5 * 40 - y) / (9.5 * 40)) + tf.nn.relu((y - 109.5 * 40) / (9.5 * 40))
        pml_tmp1 = C ** 2 * lx ** 2 * ly ** 2
        pml_tmp2 = C ** 2 * lx ** 4
        pml_tmp3 = C ** 2 * ly ** 4
        pml_tmp4 = C * (ly ** 2 - lx ** 2)
        pml_tmp5 = C * (lx ** 2 + ly ** 2)

        # u_x and u_y
        ur_x = tf.gradients(uur_pred, x)[0]
        ur_y = tf.gradients(uur_pred, y)[0]
        ui_x = tf.gradients(uui_pred, x)[0]
        ui_y = tf.gradients(uui_pred, y)[0]
        # scattered wave equations with the pml condition
        # real part
        u_r_xx = tf.gradients((1 + pml_tmp1) / (1 + pml_tmp2) * ur_x + pml_tmp4 / (1 + pml_tmp2) * ui_x, x)[0]
        u_r_yy = tf.gradients((1 + pml_tmp1) / (1 + pml_tmp3) * ur_y - pml_tmp4 / (1 + pml_tmp3) * ui_y, y)[0]
        ur_r = (1 - pml_tmp1) * (2 * math.pi * self.f) ** 2 * self.m * 1e-6 * (uur_pred + self.u0r_tf)
        ui_r = pml_tmp5 * (2 * math.pi * self.f) ** 2 * self.m * 1e-6 * (uui_pred + self.u0i_tf)
        u0r_r = (1 - pml_tmp1) * (2 * math.pi * self.f) ** 2 * (-self.m0) * 1e-6 * self.u0r_tf
        u0i_r = pml_tmp5 * (2 * math.pi * self.f) ** 2 * (-self.m0) * 1e-6 * self.u0i_tf
        f_ur_ac = u_r_xx + u_r_yy + ur_r + ui_r + u0r_r + u0i_r
        # imaginary part
        u_i_xx = tf.gradients(-pml_tmp4 / (1 + pml_tmp2) * ur_x + (1 + pml_tmp1) / (1 + pml_tmp2) * ui_x, x)[0]
        u_i_yy = tf.gradients(pml_tmp4 / (1 + pml_tmp3) * ur_y + (1 + pml_tmp1) / (1 + pml_tmp3) * ui_y, y)[0]
        ur_i = (-pml_tmp5) * (2 * math.pi * self.f) ** 2 * self.m * 1e-6 * (uur_pred + self.u0r_tf)
        ui_i = (1 - pml_tmp1) * (2 * math.pi * self.f) ** 2 * self.m * 1e-6 * (uui_pred + self.u0i_tf)
        u0r_i = (-pml_tmp5) * (2 * math.pi * self.f) ** 2 * (-self.m0) * 1e-6 * self.u0r_tf
        u0i_i = (1 - pml_tmp1) * (2 * math.pi * self.f) ** 2 * (-self.m0) * 1e-6 * self.u0i_tf
        f_ui_ac = u_i_xx + u_i_yy + ur_i + ui_i + u0r_i + u0i_i

        return uur_pred, uui_pred, f_ur_ac, f_ui_ac  # , f_ur_sc, f_ui_sc, f_r_ac_sc, f_i_ac_sc, f_r_ac_sc2, f_i_ac_sc2       #,f_ur_gradx,f_ur_grady,f_ui_gradx,f_ui_grady

    # wave equations with quadratic network
    def net_NSq(self, x, y):
        # get the output of network
        uu = self.neural_netq(x, y, self.weights, self.biases)  # output of quadratic network
        uur_pred = uu[:, 0:1]
        uui_pred = uu[:, 1:2]
        # gradients
        ur_x = tf.gradients(uur_pred, x)[0]
        ur_y = tf.gradients(uur_pred, y)[0]
        ur_xx = tf.gradients(ur_x, x)[0]
        ur_yy = tf.gradients(ur_y, y)[0]
        ui_x = tf.gradients(uui_pred, x)[0]
        ui_y = tf.gradients(uui_pred, y)[0]
        ui_xx = tf.gradients(ui_x, x)[0]
        ui_yy = tf.gradients(ui_y, y)[0]
        # scattered wave equations
        f_ur_pred = (ur_xx + ur_yy) + (2 * math.pi * self.f * 1e-3) ** 2 * self.m * uur_pred + (
                2 * math.pi * self.f * 1e-3) ** 2 * (self.m - self.m0) * self.u0r_tf
        f_ui_pred = (ui_xx + ui_yy) + (2 * math.pi * self.f * 1e-3) ** 2 * self.m * uui_pred + (
                2 * math.pi * self.f * 1e-3) ** 2 * (self.m - self.m0) * self.u0i_tf

        return uur_pred, uui_pred, f_ur_pred, f_ui_pred  # ,f_ur_gradx,f_ur_grady,f_ui_gradx,f_ui_grady

    # wave equations with pml and quadratic network
    def net_NSpmlq(self, x, y):
        self.a0 = 1.79
        self.f0 = 10
        C = self.a0 * self.f0 / self.f

        # get the output of network
        uu = self.neural_netq(x, y, self.weights, self.biases, self.wb)  # quadratic
        uur_pred = uu[:, 0:1]
        uui_pred = uu[:, 1:2]

        # pml setting
        lx = tf.nn.relu((9.5 * 20 - x) / (9.5 * 20)) + tf.nn.relu((x - 109.5 * 20) / (9.5 * 20))
        ly = tf.nn.relu((9.5 * 40 - y) / (9.5 * 40)) + tf.nn.relu((y - 109.5 * 40) / (9.5 * 40))
        pml_tmp1 = C ** 2 * lx ** 2 * ly ** 2
        pml_tmp2 = C ** 2 * lx ** 4
        pml_tmp3 = C ** 2 * ly ** 4
        pml_tmp4 = C * (ly ** 2 - lx ** 2)
        pml_tmp5 = C * (lx ** 2 + ly ** 2)

        # u_x and u_y
        ur_x = tf.gradients(uur_pred, x)[0]
        ur_y = tf.gradients(uur_pred, y)[0]
        ui_x = tf.gradients(uui_pred, x)[0]
        ui_y = tf.gradients(uui_pred, y)[0]
        # scattered wave equations with the pml condition
        # real part
        u_r_xx = tf.gradients((1 + pml_tmp1) / (1 + pml_tmp2) * ur_x + pml_tmp4 / (1 + pml_tmp2) * ui_x, x)[0]
        u_r_yy = tf.gradients((1 + pml_tmp1) / (1 + pml_tmp3) * ur_y - pml_tmp4 / (1 + pml_tmp3) * ui_y, y)[0]
        ur_r = (1 - pml_tmp1) * (2 * math.pi * self.f) ** 2 * self.m * 1e-6 * (uur_pred + self.u0r_tf)
        ui_r = pml_tmp5 * (2 * math.pi * self.f) ** 2 * self.m * 1e-6 * (uui_pred + self.u0i_tf)
        u0r_r = (1 - pml_tmp1) * (2 * math.pi * self.f) ** 2 * (-self.m0) * 1e-6 * self.u0r_tf
        u0i_r = pml_tmp5 * (2 * math.pi * self.f) ** 2 * (-self.m0) * 1e-6 * self.u0i_tf
        f_ur_ac = u_r_xx + u_r_yy + ur_r + ui_r + u0r_r + u0i_r
        # imaginary part
        u_i_xx = tf.gradients(-pml_tmp4 / (1 + pml_tmp2) * ur_x + (1 + pml_tmp1) / (1 + pml_tmp2) * ui_x, x)[0]
        u_i_yy = tf.gradients(pml_tmp4 / (1 + pml_tmp3) * ur_y + (1 + pml_tmp1) / (1 + pml_tmp3) * ui_y, y)[0]
        ur_i = (-pml_tmp5) * (2 * math.pi * self.f) ** 2 * self.m * 1e-6 * (uur_pred + self.u0r_tf)
        ui_i = (1 - pml_tmp1) * (2 * math.pi * self.f) ** 2 * self.m * 1e-6 * (uui_pred + self.u0i_tf)
        u0r_i = (-pml_tmp5) * (2 * math.pi * self.f) ** 2 * (-self.m0) * 1e-6 * self.u0r_tf
        u0i_i = (1 - pml_tmp1) * (2 * math.pi * self.f) ** 2 * (-self.m0) * 1e-6 * self.u0i_tf
        f_ui_ac = u_i_xx + u_i_yy + ur_i + ui_i + u0r_i + u0i_i

        return uur_pred, uui_pred, f_ur_ac, f_ui_ac  # , f_ur_sc, f_ui_sc, f_r_ac_sc, f_i_ac_sc, f_r_ac_sc2, f_i_ac_sc2       #,f_ur_gradx,f_ur_grady,f_ui_gradx,f_ui_grady

    # wave equations with complex velocity
    def net_NSmri(self, x, y):
        # get the output of network
        uu = self.neural_net(x, y, self.weights, self.biases, self.wb)  # output of linear network
        uur_pred = uu[:, 0:1]
        uui_pred = uu[:, 1:2]
        # gradients
        ur_x = tf.gradients(uur_pred, x)[0]
        ur_y = tf.gradients(uur_pred, y)[0]
        ur_xx = tf.gradients(ur_x, x)[0]
        ur_yy = tf.gradients(ur_y, y)[0]
        ui_x = tf.gradients(uui_pred, x)[0]
        ui_y = tf.gradients(uui_pred, y)[0]
        ui_xx = tf.gradients(ui_x, x)[0]
        ui_yy = tf.gradients(ui_y, y)[0]
        # scattered wave equations
        f_ur_ac = (ur_xx + ur_yy) + (2 * math.pi * self.f * 1e-3) ** 2 * (self.mr * uur_pred - self.mi * uui_pred) + (
                2 * math.pi * self.f * 1e-3) ** 2 * (
                              (self.mr - self.m0r) * self.u0r_tf - (self.mi - self.m0i) * self.u0i_tf)
        f_ui_ac = (ui_xx + ui_yy) + (2 * math.pi * self.f * 1e-3) ** 2 * (self.mr * uui_pred + self.mi * uur_pred) + (
                2 * math.pi * self.f * 1e-3) ** 2 * (
                              (self.mr - self.m0r) * self.u0i_tf + (self.mi - self.m0i) * self.u0r_tf)

        return uur_pred, uui_pred, f_ur_ac, f_ui_ac  # ,f_ur_gradx,f_ur_grady,f_ui_gradx,f_ui_grady

    # wave equations with complex velocity and quadratic network
    def net_NSmriq(self, x, y):
        # get the output of network
        uu = self.neural_net(x, y, self.weights, self.biases)  # output of linear network
        uur_pred = uu[:, 0:1]
        uui_pred = uu[:, 1:2]
        # gradients
        ur_x = tf.gradients(uur_pred, x)[0]
        ur_y = tf.gradients(uur_pred, y)[0]
        ur_xx = tf.gradients(ur_x, x)[0]
        ur_yy = tf.gradients(ur_y, y)[0]
        ui_x = tf.gradients(uui_pred, x)[0]
        ui_y = tf.gradients(uui_pred, y)[0]
        ui_xx = tf.gradients(ui_x, x)[0]
        ui_yy = tf.gradients(ui_y, y)[0]
        # scattered wave equations
        f_ur_ac = (ur_xx + ur_yy) + (2 * math.pi * self.f * 1e-3) ** 2 * (self.mr * uur_pred - self.mi * uui_pred) + (
                2 * math.pi * self.f * 1e-3) ** 2 * (
                              (self.mr - self.m0r) * self.u0r_tf - (self.mi - self.m0i) * self.u0i_tf)
        f_ui_ac = (ui_xx + ui_yy) + (2 * math.pi * self.f * 1e-3) ** 2 * (self.mr * uui_pred + self.mi * uur_pred) + (
                2 * math.pi * self.f * 1e-3) ** 2 * (
                              (self.mr - self.m0r) * self.u0i_tf + (self.mi - self.m0i) * self.u0r_tf)

        return uur_pred, uui_pred, f_ur_ac, f_ui_ac  # ,f_ur_gradx,f_ur_grady,f_ui_gradx,f_ui_grady

    # wave equations with complex velocity and pml
    def net_NSmripml(self, x, y):
        self.a0 = 1.79
        self.f0 = 10
        C = self.a0 * self.f0 / self.f

        # get the output of network
        uu = self.neural_net(x, y, self.weights, self.biases, self.wb)  # linear
        uur_pred = uu[:, 0:1]
        uui_pred = uu[:, 1:2]

        # pml setting
        lx = tf.nn.relu((x - 99.5 * 20) / (9.5 * 20))
        ly = tf.nn.relu((9.5 * 40 - y) / (9.5 * 40)) + tf.nn.relu((y - 109.5 * 40) / (9.5 * 40))
        pml_tmp1 = C ** 2 * lx ** 2 * ly ** 2
        pml_tmp2 = C ** 2 * lx ** 4
        pml_tmp3 = C ** 2 * ly ** 4
        pml_tmp4 = C * (ly ** 2 - lx ** 2)
        pml_tmp5 = C * (lx ** 2 + ly ** 2)

        # u_x and u_y
        ur_x = tf.gradients(uur_pred, x)[0]
        ur_y = tf.gradients(uur_pred, y)[0]
        ui_x = tf.gradients(uui_pred, x)[0]
        ui_y = tf.gradients(uui_pred, y)[0]
        # scattered wave equations with the pml condition
        # real part
        u_r_xx = tf.gradients((1 + pml_tmp1) / (1 + pml_tmp2) * ur_x + pml_tmp4 / (1 + pml_tmp2) * ui_x, x)[0]
        u_r_yy = tf.gradients((1 + pml_tmp1) / (1 + pml_tmp3) * ur_y - pml_tmp4 / (1 + pml_tmp3) * ui_y, y)[0]
        ur_r = (1 - pml_tmp1) * (2 * math.pi * self.f * 1e-3) ** 2 * (
                self.mr * (uur_pred + self.u0r_tf) - self.mi * (uui_pred + self.u0i_tf))
        ui_r = pml_tmp5 * (2 * math.pi * self.f * 1e-3) ** 2 * (
                self.mr * (uui_pred + self.u0i_tf) + self.mi * (uur_pred + self.u0r_tf))
        u0r_r = (1 - pml_tmp1) * (2 * math.pi * self.f * 1e-3) ** 2 * (-self.m0r * self.u0r_tf + self.m0i * self.u0i_tf)
        u0i_r = pml_tmp5 * (2 * math.pi * self.f * 1e-3) ** 2 * (-self.m0r * self.u0i_tf - self.m0i * self.u0r_tf)
        f_ur_ac = u_r_xx + u_r_yy + ur_r + ui_r + u0r_r + u0i_r
        # imaginary part
        u_i_xx = tf.gradients(-pml_tmp4 / (1 + pml_tmp2) * ur_x + (1 + pml_tmp1) / (1 + pml_tmp2) * ui_x, x)[0]
        u_i_yy = tf.gradients(pml_tmp4 / (1 + pml_tmp3) * ur_y + (1 + pml_tmp1) / (1 + pml_tmp3) * ui_y, y)[0]
        ur_i = (-pml_tmp5) * (2 * math.pi * self.f * 1e-3) ** 2 * (
                self.mr * (uur_pred + self.u0r_tf) - self.mi * (uui_pred + self.u0i_tf))
        ui_i = (1 - pml_tmp1) * (2 * math.pi * self.f * 1e-3) ** 2 * (
                self.mr * (uui_pred + self.u0i_tf) + self.mi * (uur_pred + self.u0r_tf))
        u0r_i = (-pml_tmp5) * (2 * math.pi * self.f * 1e-3) ** 2 * (-self.m0r * self.u0r_tf + self.m0i * self.u0i_tf)
        u0i_i = (1 - pml_tmp1) * (2 * math.pi * self.f * 1e-3) ** 2 * (-self.m0r * self.u0i_tf - self.m0i * self.u0r_tf)
        f_ui_ac = u_i_xx + u_i_yy + ur_i + ui_i + u0r_i + u0i_i

        return uur_pred, uui_pred, f_ur_ac, f_ui_ac  #

    # wave equations with complex velocity and pml and quadratic network
    def net_NSmripmlq(self, x, y):
        self.a0 = 1.79
        self.f0 = 10
        C = self.a0 * self.f0 / self.f

        # get the output of network
        uu = self.neural_netqw5qw2all(x, y, self.weights, self.biases, self.wb)  # linear
        uur_pred = uu[:, 0:1]
        uui_pred = uu[:, 1:2]

        # pml setting
        lx = tf.nn.relu((x - 99.5 * 20) / (9.5 * 20))
        ly = tf.nn.relu((9.5 * 40 - y) / (9.5 * 40)) + tf.nn.relu((y - 109.5 * 40) / (9.5 * 40))
        pml_tmp1 = C ** 2 * lx ** 2 * ly ** 2
        pml_tmp2 = C ** 2 * lx ** 4
        pml_tmp3 = C ** 2 * ly ** 4
        pml_tmp4 = C * (ly ** 2 - lx ** 2)
        pml_tmp5 = C * (lx ** 2 + ly ** 2)

        # u_x and u_y
        ur_x = tf.gradients(uur_pred, x)[0]
        ur_y = tf.gradients(uur_pred, y)[0]
        ui_x = tf.gradients(uui_pred, x)[0]
        ui_y = tf.gradients(uui_pred, y)[0]
        # scattered wave equations with the pml condition
        # real part
        u_r_xx = tf.gradients((1 + pml_tmp1) / (1 + pml_tmp2) * ur_x + pml_tmp4 / (1 + pml_tmp2) * ui_x, x)[0]
        u_r_yy = tf.gradients((1 + pml_tmp1) / (1 + pml_tmp3) * ur_y - pml_tmp4 / (1 + pml_tmp3) * ui_y, y)[0]
        ur_r = (1 - pml_tmp1) * (2 * math.pi * self.f * 1e-3) ** 2 * (
                self.mr * (uur_pred + self.u0r_tf) - self.mi * (uui_pred + self.u0i_tf))
        ui_r = pml_tmp5 * (2 * math.pi * self.f * 1e-3) ** 2 * (
                self.mr * (uui_pred + self.u0i_tf) + self.mi * (uur_pred + self.u0r_tf))
        u0r_r = (1 - pml_tmp1) * (2 * math.pi * self.f * 1e-3) ** 2 * (-self.m0r * self.u0r_tf + self.m0i * self.u0i_tf)
        u0i_r = pml_tmp5 * (2 * math.pi * self.f * 1e-3) ** 2 * (-self.m0r * self.u0i_tf - self.m0i * self.u0r_tf)
        f_ur_ac = u_r_xx + u_r_yy + ur_r + ui_r + u0r_r + u0i_r
        # imaginary part
        u_i_xx = tf.gradients(-pml_tmp4 / (1 + pml_tmp2) * ur_x + (1 + pml_tmp1) / (1 + pml_tmp2) * ui_x, x)[0]
        u_i_yy = tf.gradients(pml_tmp4 / (1 + pml_tmp3) * ur_y + (1 + pml_tmp1) / (1 + pml_tmp3) * ui_y, y)[0]
        ur_i = (-pml_tmp5) * (2 * math.pi * self.f * 1e-3) ** 2 * (
                self.mr * (uur_pred + self.u0r_tf) - self.mi * (uui_pred + self.u0i_tf))
        ui_i = (1 - pml_tmp1) * (2 * math.pi * self.f * 1e-3) ** 2 * (
                self.mr * (uui_pred + self.u0i_tf) + self.mi * (uur_pred + self.u0r_tf))
        u0r_i = (-pml_tmp5) * (2 * math.pi * self.f * 1e-3) ** 2 * (-self.m0r * self.u0r_tf + self.m0i * self.u0i_tf)
        u0i_i = (1 - pml_tmp1) * (2 * math.pi * self.f * 1e-3) ** 2 * (-self.m0r * self.u0i_tf - self.m0i * self.u0r_tf)
        f_ui_ac = u_i_xx + u_i_yy + ur_i + ui_i + u0r_i + u0i_i

        return uur_pred, uui_pred, f_ur_ac, f_ui_ac  #

    # wave equations with complex velocity and pml and quadratic network
    def Freesurface(self, x, y):
        uu = self.neural_netqw5qw2all(x, y, self.weights, self.biases, self.wb)
        urb = uu[:, 0:1]
        uib = uu[:, 1:2]
        return urb,uib  #

    def callback(self, loss):
        print('Loss: %.3e' % loss)

    def train(self, nIter):
        # feed the input
        # tf_dict_all = {self.x_tf: self.x, self.y_tf: self.y, self.v_tf: self.v, self.vsm_tf: self.vsm,
        #            self.u0r_tf: self.u0r, self.u0i_tf: self.u0i}
        # save initialed network parameters
        for lwb in range(len(self.weights)):
            wpara = self.sess.run(self.weights[lwb])
            bpara = self.sess.run(self.biases[lwb])
            np.save("./test2/w" + str(lwb) + ".npy", wpara)
            np.save("./test2/b" + str(lwb) + ".npy", bpara)

        # training loop
        start_time = time.time()
        loss_list = []
        for it in range(nIter):
            print(it)
            N_train = 6600
            idx = np.random.choice(Ny * Nx, N_train, replace=False)
            idx_sum = idx
            x_batch = self.x[idx_sum, :]
            y_batch = self.y[idx_sum, :]
            # xb_batch = self.x[10:110, :]
            # yb_batch = self.y[10:110, :]
            v_batch = self.v[idx_sum, :]
            u0r_batch = self.u0r[idx_sum, :]
            u0i_batch = self.u0i[idx_sum, :]
            # tf_dict = {self.x_tf: x_batch, self.y_tf: y_batch, self.v_tf: v_batch,
            #            self.u0r_tf: u0r_batch, self.u0i_tf: u0i_batch, self.xb_tf: xb_batch, self.yb_tf: yb_batch}
            tf_dict = {self.x_tf: x_batch, self.y_tf: y_batch, self.v_tf: v_batch,
                       self.u0r_tf: u0r_batch, self.u0i_tf: u0i_batch}
            # update the parameters
            self.sess.run(self.train_op_Adam, tf_dict)
            loss_value = self.sess.run(self.loss, tf_dict)
            loss_list.append(loss_value)
            lost_np = np.array(loss_list)
            # Print
            if it % 1000 == 0:
                ur, ui = self.predict(self.x, self.y)
                fig = plt.figure()
                durplt = np.reshape(ui, (Nx, Ny))
                plt.imshow(durplt, cmap='bwr')
                plt.colorbar()
                plt.savefig('./test2/ui' + str(it) + '.png')
                plt.close(fig)
                # np.save("./test2/ui" + str(it) + ".npy", durplt)
                fig = plt.figure()
                durplt = np.reshape(ur, (Nx, Ny))
                plt.imshow(durplt, cmap='bwr')
                plt.colorbar()
                plt.savefig('./test2/ur' + str(it) + '.png')
                plt.close(fig)
                # np.save("./test2/ur" + str(it) + ".npy", durplt)
                start_time = time.time()
                for lwbfine in range(len(self.weights)):
                    wpara = self.sess.run(self.weights[lwbfine])
                    bpara = self.sess.run(self.biases[lwbfine])
                    np.save("./test2/wadam" + str(lwbfine) + ".npy", wpara)
                    np.save("./test2/badam" + str(lwbfine) + ".npy", bpara)
                wbpara = self.sess.run(self.wb)
                np.save("./test2/wbadam.npy", wbpara)
            if it % 25000 == 0:
                for lwbfine in range(len(self.weights)):
                    wpara = self.sess.run(self.weights[lwbfine])
                    bpara = self.sess.run(self.biases[lwbfine])
                    np.save("./test2/wadam" + str(lwbfine) + str(it) + ".npy", wpara)
                    np.save("./test2/badam" + str(lwbfine) + str(it) + ".npy", bpara)
                wbpara = self.sess.run(self.wb)
                np.save("./test2/wbadam.npy", wbpara)
        fig = plt.figure()
        plt.plot(lost_np)
        plt.savefig('./test2/loss.png')
        plt.close(fig)
        np.save("test2/loss.npy", lost_np)

    def predict(self, x, y):

        tf_dict = {self.x_tf: x, self.y_tf: y}
        ur_star = self.sess.run(self.urnet, tf_dict)
        ui_star = self.sess.run(self.uinet, tf_dict)

        return ur_star, ui_star


if __name__ == "__main__":

    # layers = [2,40,40,40,40,40,40,40,40,40,40,40,40,40,40,40,40,40,40,40,40, 2]  # the number of neure in each layer
    # layers = [2,60,60,60,60,60,60,60,60,60,60, 2]
    layers = [2, 80, 80, 80, 80, 80, 80, 80, 80, 2]  # p1
    # layers = [2, 14, 80, 80, 80, 80, 80, 80, 80, 2]              #p2
    # layers = [2, 14, 14, 80, 80, 80, 80, 80, 80, 2]              #p3
    # layers = [2, 14, 14, 14, 14, 14, 14, 14, 14, 2]              #pn
    # layers = [5, 128, 128, 64, 64, 32, 32, 16, 16, 8, 8, 2]

    marmousi = scipy.io.loadmat('./loaddata/v_deeper.mat')
    vma = marmousi['v_deeper']
    u0fsmat = scipy.io.loadmat('./loaddata/u0_deeper.mat')
    u0 = u0fsmat['u0_deeper']

    # u0fsmat = scipy.io.loadmat('./loaddata/u0fsmri.mat')
    # u0fs_5 = u0fsmat['u0fsmri']
    # u0fs_5 = u0fs_5.transpose(2, 0, 1)
    # u0fs_1 = np.zeros((np.size(u0fs_5, 0), 120, 120), dtype='complex_')
    # for ii in range(0, 120):
    #     for ij in range(0, 120):
    #         u0fs_1[:, ii, ij] = u0fs_5[:, ii * 5 + 2, ij * 5 + 2]
    # u0 = u0fs_1[50, :, :]

    ###############################################
    ############### PML or Not ####################
    ###############################################
    # Without PML
    # Nx = 100
    # Ny = 100
    # vv=vma[10:110,10:110]

    # With PML
    Nx = 110
    Ny = 120
    vv = vma

    ###############################################
    ########## x, y, w(or s), v, u0 ###############
    ###############################################
    # define the input x y f v u0
    # x y f
    yy = 40 * np.tile(np.arange(Ny), [1, Nx]).T
    xx = np.tile(0, [1, Ny])
    for i in range(1, Nx):
        x = np.tile(i, [1, Ny])
        xx = np.hstack((xx, x))
    xx = 20 * xx.T
    #  v
    fig = plt.figure()
    plt.imshow(vv)
    plt.colorbar()
    plt.savefig('./test2/vmodel')
    plt.close(fig)
    vv = np.reshape(vv, [Ny * Nx, 1])
    # u0
    u0 = u0.reshape((Nx * Ny, 1))
    # # u0 hankle
    # loca = ((xx-20*4)**2+(yy-40*49)**2)**0.5
    # u0fs = 210*1j/4*ss.hankel2(0,2*math.pi*freqs/1500*loca)

    # use all of the points
    x_train = xx
    y_train = yy
    v_train = vv
    um0r_train = np.real(u0)
    um0i_train = np.imag(u0)

    ###############################################
    ############## Training #######################
    ###############################################
    model = PhysicsInformedNN(xx, yy, layers, v_train, um0r_train, um0i_train)
    model.train(150001)

