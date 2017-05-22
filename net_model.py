import lasagne.layers
import lasagne.layers.shape
import lasagne.layers.dnn 
import lasagne.nonlinearities
import lasagne.init
import pickle
import theano.tensor as T
import theano

import ctc_cost

class Model:

    def __init__(self, num_classes, clip_length, img_channels, batch_size, num_batch_eval, num_steps):

        self.num_classes = num_classes
        self.clip_length = clip_length
        self.img_channels = img_channels
        self.batch_size = batch_size
        self.num_batch_eval = num_batch_eval
        self.num_steps = num_steps
        self.net = {}

        self._set_net()

    def _set_net(self):

        if_flip_filters = False
        num_rnn_units = 256
        GRAD_CLIP = 100             # for LSTM layer, all gradients above this will be clipped
        H_net_input = 112           # 224  # 112
        W_net_input = 112           # 224  # 112

        # ------------------ input layer group ----------------
        self.net['input'] = lasagne.layers.InputLayer((None, self.img_channels, self.clip_length, H_net_input, W_net_input))
        self.net['mask'] = lasagne.layers.InputLayer((None, self.num_steps))

        # ----------------- 1st CNN layer group ---------------
        self.net['conv1a'] = lasagne.layers.dnn.Conv3DDNNLayer(self.net['input'],
                                            64,
                                            (3,3,3), 
                                            pad=1,
                                            nonlinearity=lasagne.nonlinearities.rectify,
                                            flip_filters=if_flip_filters, 
                                            W=lasagne.init.Normal(std=0.01), 
                                            b=lasagne.init.Constant(0.))

        self.net['pool1']  = lasagne.layers.dnn.MaxPool3DDNNLayer(self.net['conv1a'],
                                               pool_size=(1,2,2),
                                               stride=(1,2,2))

        # ----------------- 2nd CNN layer group --------------
        self.net['conv2a'] = lasagne.layers.dnn.Conv3DDNNLayer(self.net['pool1'], 
                                            128, 
                                            (3,3,3), 
                                            pad=1,
                                            nonlinearity=lasagne.nonlinearities.rectify,
                                            flip_filters=if_flip_filters, 
                                            W=lasagne.init.Normal(std=0.01), 
                                            b=lasagne.init.Constant(1.))

        self.net['pool2']  = lasagne.layers.dnn.MaxPool3DDNNLayer(self.net['conv2a'],
                                               pool_size=(2,2,2),
                                               stride=(2,2,2))

        # ----------------- 3rd CNN layer group --------------
        self.net['conv3a'] = lasagne.layers.dnn.Conv3DDNNLayer(self.net['pool2'], 
                                            256, 
                                            (3,3,3), 
                                            pad=1,
                                            nonlinearity=lasagne.nonlinearities.rectify,
                                            flip_filters=if_flip_filters, 
                                            W=lasagne.init.Normal(std=0.01), 
                                            b=lasagne.init.Constant(1.))

        self.net['conv3b'] = lasagne.layers.dnn.Conv3DDNNLayer(self.net['conv3a'], 
                                            256, 
                                            (3,3,3), 
                                            pad=1,
                                            nonlinearity=lasagne.nonlinearities.rectify,
                                            flip_filters=if_flip_filters, 
                                            W=lasagne.init.Normal(std=0.01), 
                                            b=lasagne.init.Constant(1.))

        self.net['pool3']  = lasagne.layers.dnn.MaxPool3DDNNLayer(self.net['conv3b'],
                                               pool_size=(2,2,2),
                                               stride=(2,2,2))

        # ----------------- 4th CNN layer group --------------
        self.net['conv4a'] = lasagne.layers.dnn.Conv3DDNNLayer(self.net['pool3'], 
                                            512, 
                                            (3,3,3), 
                                            pad=1,
                                            nonlinearity=lasagne.nonlinearities.rectify,
                                            flip_filters=if_flip_filters, 
                                            W=lasagne.init.Normal(std=0.01), 
                                            b=lasagne.init.Constant(1.))

        self.net['conv4b'] = lasagne.layers.dnn.Conv3DDNNLayer(self.net['conv4a'], 
                                            512,
                                            (3,3,3), 
                                            pad=1,
                                            nonlinearity=lasagne.nonlinearities.rectify,
                                            flip_filters=if_flip_filters, 
                                            W=lasagne.init.Normal(std=0.01), 
                                            b=lasagne.init.Constant(1.))

        self.net['pool4']  = lasagne.layers.dnn.MaxPool3DDNNLayer(self.net['conv4b'],
                                               pool_size=(2,2,2),
                                               stride=(2,2,2))

        # ----------------- 5th CNN layer group --------------
        self.net['conv5a'] = lasagne.layers.dnn.Conv3DDNNLayer(self.net['pool4'], 
                                            512, 
                                            (3,3,3), 
                                            pad=1,
                                            nonlinearity=lasagne.nonlinearities.rectify,
                                            flip_filters=if_flip_filters, 
                                            W=lasagne.init.Normal(std=0.01), 
                                            b=lasagne.init.Constant(1.))

        self.net['conv5b'] = lasagne.layers.dnn.Conv3DDNNLayer(self.net['conv5a'], 
                                            512, 
                                            (3,3,3), 
                                            pad=1,
                                            nonlinearity=lasagne.nonlinearities.rectify,
                                            flip_filters=if_flip_filters, 
                                            W=lasagne.init.Normal(std=0.01), 
                                            b=lasagne.init.Constant(1.))

        # -------------------- 6th layer group ----------------
        # We need a padding layer, as C3D only pads on the right, which cannot be done with a theano pooling layer
        self.net['pad'] = lasagne.layers.shape.PadLayer(self.net['conv5b'],
                                   width=[(0,1),(0,1)], 
                                   batch_ndim=3)

        self.net['pool5'] = lasagne.layers.dnn.MaxPool3DDNNLayer(self.net['pad'],
                                              pool_size=(2,2,2),
                                              pad=(0,0,0),
                                              stride=(2,2,2))

        self.net['fc6-1'] = lasagne.layers.DenseLayer(self.net['pool5'], 
                                      num_units=4096, 
                                      nonlinearity=lasagne.nonlinearities.rectify,
                                      W=lasagne.init.Normal(std=0.005), 
                                      b=lasagne.init.Constant(1.))

        self.net['drop6'] = lasagne.layers.DropoutLayer(self.net['fc6-1'], 
                                         p=0.5)

        # ----------------- 7th lstm layer group -------------
        dim_fc6 = self.net['drop6'].output_shape
        self.net['fc6_resize'] = lasagne.layers.ReshapeLayer(self.net['drop6'], 
                                              (-1, self.num_steps)+dim_fc6[1:])

        # if use bidirectional RNN
        use_biRNN = False
        if use_biRNN:
            self.net['lstm7_forward'] = lasagne.layers.LSTMLayer(self.net['fc6_resize'], 
                                                  num_units=num_rnn_units, 
                                                  only_return_final=True, 
                                                  grad_clipping=GRAD_CLIP,
                                                  nonlinearity=lasagne.nonlinearities.tanh, 
                                                  mask_input=self.net['mask'])

            self.net['lstm7_backward'] = lasagne.layers.LSTMLayer(self.net['fc6_resize'], 
                                                   num_units=num_rnn_units, 
                                                   only_return_final=True, 
                                                   backwards=True,
                                                   grad_clipping=GRAD_CLIP, 
                                                   nonlinearity=lasagne.nonlinearities.tanh, 
                                                   mask_input=self.net['mask'])

            self.net['lstm7'] = lasagne.layers.ConcatLayer([self.net['lstm7_forward'], 
                                                           self.net['lstm7_backward']])
        else:
            self.net['lstm7'] = lasagne.layers.LSTMLayer(self.net['fc6_resize'], 
                                          num_units=num_rnn_units, 
                                          unroll_scan=True, 
                                          only_return_final=True, 
                                          grad_clipping=GRAD_CLIP,
                                          nonlinearity=lasagne.nonlinearities.tanh, 
                                          cell_init=lasagne.init.Orthogonal(), 
                                          hid_init=lasagne.init.Orthogonal(), 
                                          learn_init=True, 
                                          mask_input=self.net['mask'])

        # ------------------- 8th layer group ---------------
        self.net['lstm7_dropout'] = lasagne.layers.DropoutLayer(self.net['lstm7'], p=0.5)


        self.net['fc8-1'] = lasagne.layers.DenseLayer(self.net['lstm7_dropout'], 
                                       num_units=self.num_classes, 
                                       nonlinearity=None, 
                                       W=lasagne.init.Normal(std=0.01), 
                                       b=lasagne.init.Constant(0.))

        self.net['prob'] = lasagne.layers.NonlinearityLayer(self.net['fc8-1'], 
                                             lasagne.nonlinearities.softmax)


    def _set_model_param(self, Dir_features):
        # you could also use the model trained by caffe since you can read parameters with pycaffe 
        # if the last model is trained with caffe model, then used_caffe_model should be True

        used_last_model = False  
        if used_last_model:

            model_last = pickle.load(open(Dir_features + 'c3d_last_model.pkl'))
            print('the last time trained c3d len(model_last):', len(model_last))
            print('inherit the parameters from the model trained last time')

            lasagne.layers.set_all_param_values(self.net['prob'], model_last)

        used_pretrained_c3d = True
        if used_pretrained_c3d:

            with open(Dir_features+'c3d_pretrained.pkl', 'rb') as f:
                model_pretrained_c3d = pickle.load(f)
            print('the pretrained c3d model len(model_pretrained_c3d):', len(model_pretrained_c3d))

            # notice that if you add rnn_spn between conv layers, the structure of C3D is also changed
            # by now, since the rnn_spn is add to the last conv layer, the problem can be simplified:
            lasagne.layers.set_all_param_values(self.net['fc6-1'], model_pretrained_c3d[:-4], trainable=True)

        used_caffe_model = True
        if used_caffe_model and not used_pretrained_c3d and not used_last_model:

            model_file = Dir_features + 'c3d_model.pkl'
            with open(model_file) as f:
                print('Load pretrained weights from %s...' % model_file)
                model = pickle.load(f)
            print('Set the weights...')

            # notice that if you add rnn_spn between conv layers, the structure of C3D is also changed
            # by now, since the rnn_spn is add to the last conv layer, the problem can be simplified:
            lasagne.layers.set_all_param_values(self.net['fc6-1'], model[:-4], trainable=True)


    def build_model(self, Dir_features, args):

        self._set_model_param(Dir_features)

        # try to scale the gradients on the level of parameters like caffe
        # by now only change the code with sgd
        scale_grad = True
        scale_l2_w = False

        TOL = 1e-5

        sym_y = T.imatrix()

        # W is regularizable, b is not regularizable (correspondence with caffe)
        if scale_grad:
            self.net['conv1a'].b.tag.grad_scale = 2
            self.net['conv2a'].b.tag.grad_scale = 2
            self.net['conv3a'].b.tag.grad_scale = 2
            self.net['conv3b'].b.tag.grad_scale = 2
            self.net['conv4a'].b.tag.grad_scale = 2
            self.net['conv4b'].b.tag.grad_scale = 2
            self.net['conv5a'].b.tag.grad_scale = 2
            self.net['conv5b'].b.tag.grad_scale = 2
            self.net['fc6-1'].b.tag.grad_scale = 2
            self.net['fc8-1'].W.tag.grad_scale = 10
            self.net['fc8-1'].b.tag.grad_scale = 20

        output_train = lasagne.layers.get_output(self.net['prob'], deterministic=False)
        output_eval = lasagne.layers.get_output(self.net['prob'], deterministic=True)

        ##############
        # compute cost
        ##############
        # compute the cost for training
        output_flat = T.reshape(output_train, (self.batch_size, self.clip_length, self.num_classes))
        cost = T.mean(ctc_cost.cost(output_flat+TOL, sym_y))

        # maybe it is necessary to add l2_penalty to the cost
        regularizable_params = lasagne.layers.get_all_params(self.net['prob'], regularizable=True)
        l2_w = 0.0005
        all_layers = lasagne.layers.get_all_layers(self.net['prob'])
        l2_penalty = lasagne.regularization.regularize_layer_params(all_layers, lasagne.regularization.l2)*l2_w
        cost += l2_penalty

        # compute the cost for evaluation
        output_eval_flat = T.reshape(output_eval, (self.num_batch_eval, self.clip_length, self.num_classes))
        cost_eval = T.mean(ctc_cost.cost(output_eval_flat+TOL, sym_y))

        trainable_params = lasagne.layers.get_all_params(self.net['prob'], trainable=True)

        sh_lr = theano.shared(lasagne.utils.floatX(args.lr))

        ##################################################################
        # try to scale the gradients on the level of parameters like caffe
        # by now only change the code with sgd
        ##################################################################
        if scale_grad:
            grads = theano.grad(cost, trainable_params)
            for idx, param in enumerate(trainable_params):
                grad_scale = getattr(trainable_params, 'grad_scale', 1)
                if grad_scale != 1:
                    grads[idx] *= grad_scale
        
        #################
        # compute updates
        #################
        # adam works with lr 0.001
        if args.optimizer == 'rmsprop':
            updates_opt = lasagne.updates.rmsprop(cost, trainable_params, learning_rate=sh_lr)
            updates = lasagne.updates.apply_momentum(updates_opt, trainable_params, momentum=0.9)

        elif args.optimizer == 'adam':
            updates_opt = lasagne.updates.adam(cost, trainable_params, learning_rate=sh_lr)
            updates = lasagne.updates.apply_momentum(updates_opt, trainable_params, momentum=0.9)

        elif args.optimizer == 'sgd':
            # Stochastic Gradient Descent (SGD) with momentum
            if scale_grad:
                updates = lasagne.updates.momentum(grads, trainable_params, learning_rate=sh_lr, momentum=0.9)
            else:
                updates = lasagne.updates.momentum(cost, trainable_params, learning_rate=sh_lr, momentum=0.9)

        elif args.optimizer == 'adadelta':
            updates_opt = lasagne.updates.adadelta(cost, trainable_params, learning_rate=sh_lr)
            updates = lasagne.updates.apply_momentum(updates_opt, trainable_params, momentum=0.9)

        elif args.optimizer == 'adagrad':
            updates_opt = lasagne.updates.adagrad(cost, trainable_params, learning_rate=sh_lr)
            updates = lasagne.updates.apply_momentum(updates_opt, trainable_params, momentum=0.9)
        
        #############################
        # set train and eval function
        #############################
        f_train = theano.function([self.net['input'].input_var, sym_y, self.net['mask'].input_var], [cost, output_train], updates=updates)
        f_eval = theano.function([self.net['input'].input_var, sym_y, self.net['mask'].input_var], [cost_eval, output_eval])
        
        return f_train, f_eval