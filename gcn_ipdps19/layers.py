import tensorflow as tf
from gcn_ipdps19.inits import glorot,zeros,trained
from gcn_ipdps19.mkl_wrapper import *


# global unique layer ID dictionary for layer name assignment
_LAYER_UIDS = {}
FLAGS=tf.app.flags.FLAGS

def get_layer_uid(layer_name=''):
    """Helper function, assigns unique layer IDs."""
    if layer_name not in _LAYER_UIDS:
        _LAYER_UIDS[layer_name] = 1
        return 1
    else:
        _LAYER_UIDS[layer_name] += 1
        return _LAYER_UIDS[layer_name]

class Layer:
    """Base layer class. Defines basic API for all layer objects.
    Implementation inspired by keras (http://keras.io).
    # Properties
        name: String, defines the variable scope of the layer.
        logging: Boolean, switches Tensorflow histogram logging on/off

    # Methods
        _call(inputs): Defines computation graph of layer
            (i.e. takes input, returns output)
        __call__(inputs): Wrapper for _call()
        _log_vars(): Log all variables
    """

    def __init__(self, **kwargs):
        allowed_kwargs = {'name', 'logging'}
        for kwarg in kwargs.keys():
            assert kwarg in allowed_kwargs, 'Invalid keyword argument: ' + kwarg
        name = kwargs.get('name')
        if not name:
            layer = self.__class__.__name__.lower()
            name = layer + '_' + str(get_layer_uid(layer))
        self.name = name
        self.vars = {}
        logging = kwargs.get('logging', False)
        self.logging = logging

    def _call(self, inputs):
        return inputs

    def __call__(self, inputs, is_act=True):
        with tf.name_scope(self.name):
            if self.logging:
                tf.summary.histogram(self.name + '/inputs', inputs)
            outputs = self._call(inputs, is_act=is_act)
            if self.logging:
                tf.summary.histogram(self.name + '/outputs', outputs)
        return outputs

    def _log_vars(self):
        for var in self.vars:
            tf.summary.histogram(self.name + '/vars/' + var, self.vars[var])


class Dense(Layer):
    """Dense layer."""
    def __init__(self, dim_in, dim_out, weight_decay, dropout=0.,
                 act=tf.nn.relu, bias=True,  **kwargs):
        super(Dense, self).__init__(**kwargs)
        self.dropout = dropout
        self.act = act
        self.bias = bias
        self.dim_in = dim_in
        self.dim_out = dim_out
        self.weight_decay = weight_decay

        with tf.variable_scope(self.name + '_vars'):
            self.vars['weights'] = tf.get_variable('weights', shape=(dim_in, dim_out),
                                     dtype=tf.float32,
                                     initializer=tf.contrib.layers.xavier_initializer(),
                                     regularizer=tf.contrib.layers.l2_regularizer(self.weight_decay))
            if self.bias:
                self.vars['bias'] = zeros([dim_out],name='bias')
        if self.logging:
            self._log_vars()

    def _call(self, inputs, is_act=True):
        x = tf.nn.dropout(inputs, 1-self.dropout)
        if FLAGS.mkl:
            output = tf_sgemm(x, self.vars['weights'])
        else:
            output=tf.matmul(x,self.vars['weights'])
        if self.bias:
            output += self.vars['bias']
        return self.act(output)



class Mean_Aggregator(Layer):
    """
    Aggregates via mean followed by matmul and non-linearity.
    """

    def __init__(self, dim_in, dim_out, neigh_dim_in=None,
            dropout=0., bias=False, act=tf.nn.relu, **kwargs):
        super(Mean_Aggregator, self).__init__(**kwargs)

        self.dropout = dropout
        self.bias = bias
        self.act = act

        assert neigh_dim_in is None
        if neigh_dim_in is None:
            neigh_dim_in = dim_in

        with tf.variable_scope(self.name + '_vars'):
            self.vars['neigh_weights'] = glorot([neigh_dim_in,dim_out], name='neigh_weights')
            self.vars['self_weights'] = glorot([dim_in,dim_out], name='self_weights')
            if self.bias:
                self.vars['bias'] = zeros([self.dim_out], name='bias')

        if self.logging:
            self._log_vars()

        self.dim_in = dim_in
        self.dim_out = dim_out

    def _call(self, inputs, is_act=True):
        neigh_vecs, self_vecs, adj_norm = inputs
        #neigh_vecs = tf.nn.dropout(neigh_vecs, 1-self.dropout)
        #self_vecs = tf.nn.dropout(self_vecs, 1-self.dropout)
        if FLAGS.mkl:
            neigh_means = tf_scoomm(adj_norm, neigh_vecs)
            from_neighs = tf_sgemm(neigh_means, self.vars['neigh_weights'])
            from_self = tf_sgemm(self_vecs, self.vars["self_weights"])
        else:
            neigh_means=tf.sparse_tensor_dense_matmul(adj_norm,neigh_vecs)
            from_neighs=tf.matmul(neigh_means,self.vars['neigh_weights'])
            from_self=tf.matmul(self_vecs,self.vars['self_weights'])
        output = tf.concat([from_self, from_neighs], axis=1)
        if self.bias:
            output += self.vars['bias']
        return self.act(output) if is_act else output
