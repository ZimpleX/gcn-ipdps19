import tensorflow as tf
from gcn_ipdps19.inits import *
import gcn_ipdps19.layers as layers
import pdb




class GCN_Subgraph:

    def __init__(self, num_classes, placeholders, features,
            dims, train_params, loss='softmax',  **kwargs):
        '''
        Args:
            - placeholders: Stanford TensorFlow placeholder object.
            - features: Numpy array with node features.
            - adj: Numpy array with adjacency lists (padded with random re-samples)
            - degrees: Numpy array with node degrees.
            - sigmoid_loss: Set to true if nodes can belong to multiple classes
        '''
        self.aggregator_cls = layers.Mean_Aggregator
        self.lr = train_params['lr']
        self.node_subgraph = placeholders['node_subgraph']
        self.num_layers = len(dims)
        self.weight_decay = train_params['weight_decay']
        self.adj_subgraph  = placeholders['adj_subgraph']
        self.features = tf.Variable(tf.constant(features, dtype=tf.float32), trainable=False)
        self.num_classes = num_classes
        self.sigmoid_loss = (loss=='sigmoid')
        self.dims = [0 if features is None else features.shape[1]] + list(dims)
        self.placeholders = placeholders

        self.optimizer = tf.train.AdamOptimizer(learning_rate=self.lr,\
                    beta1=0.9,beta2=0.999,epsilon=1e-8,use_locking=False,name='Adam')
        self.loss = 0
        self.opt_op = None
        self.build()


    def build(self):
        """
        Build the sample graph with adj info in self.sample()
        directly feed the sampled support vectors to tf placeholder
        """
        self.aggregators = self.get_aggregators()
        self.outputs = self.aggregate_subgraph(self.node_subgraph,\
                            self.features,self.aggregators,adjs=self.adj_subgraph)
        #####################
        # [z]: OUPTUT LAYER #
        #####################
        self.outputs = tf.nn.l2_normalize(self.outputs, 1)
        self.node_pred = layers.Dense(2*self.dims[-1], self.num_classes, self.weight_decay,
                dropout=self.placeholders['dropout'], act=lambda x:x)
        self.node_preds = self.node_pred(self.outputs)

        #####################
        # [z]: BACK PROP    #
        #####################
        self._loss()
        grads_and_vars = self.optimizer.compute_gradients(self.loss)
        clipped_grads_and_vars = [(tf.clip_by_value(grad, -5.0, 5.0) if grad is not None else None, var)
                for grad, var in grads_and_vars]
        self.grad, _ = clipped_grads_and_vars[0]
        self.opt_op = self.optimizer.apply_gradients(clipped_grads_and_vars)
        self.preds = self.predict()


    def _loss(self):
        for aggregator in self.aggregators:
            for var in aggregator.vars.values():
                self.loss += self.weight_decay * tf.nn.l2_loss(var)
        for var in self.node_pred.vars.values():
            self.loss += self.weight_decay * tf.nn.l2_loss(var)

        f_loss = tf.nn.sigmoid_cross_entropy_with_logits if self.sigmoid_loss\
                                else tf.nn.softmax_cross_entropy_with_logits
        self.loss += tf.reduce_mean(f_loss(logits=self.node_preds,labels=self.placeholders['labels']))
        tf.summary.scalar('loss', self.loss)

    def predict(self):
        return tf.nn.sigmoid(self.node_preds) if self.sigmoid_loss \
                else tf.nn.softmax(self.node_preds)


    def get_aggregators(self,name=None):
        aggregators = []
        for layer in range(self.num_layers):
            dim_mult = (layer!=0) and 2 or 1
            aggregator = self.aggregator_cls(dim_mult*self.dims[layer], self.dims[layer+1],
                    dropout=self.placeholders['dropout'],name=name)
            aggregators.append(aggregator)
        return aggregators


    def aggregate_subgraph(self, samples, input_features, aggregators, batch_size=None, name=None, adjs=None):
        hidden = tf.nn.embedding_lookup(input_features, samples)
        for layer in range(self.num_layers):
            dim_mult = (layer != 0) and 2 or 1
            # last layer does not have ReLU activation function
            is_act = False if layer==self.num_layers-1 else True
            hidden = aggregators[layer]((hidden, hidden, adjs), is_act=is_act)
        return hidden

