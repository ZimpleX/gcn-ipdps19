import math
from gcn_ipdps19.inits import *
from gcn_ipdps19.graph_samplers import *
import tensorflow as tf
import scipy.sparse as sp

import numpy as np
import time
import multiprocessing as mp

import pdb

# import warnings
# warnings.filterwarnings('error')
# np.seterr(divide='raise')

np.random.seed(123)


############################
# FOR SUPERVISED MINIBATCH #
############################
class NodeMinibatchIterator(object):
    """
    This minibatch iterator iterates over nodes for supervised learning.
    """

    def __init__(self, adj_full, adj_train, role, class_arr, placeholders, **kwargs):
        """
        role:       array of string (length |V|)
                    storing role of the node ('tr'/'va'/'te')
        class_arr: array of float (shape |V|xf)
                    storing initial feature vectors
        """
        self.num_proc = 40
        self.node_train = np.array(role['tr'])
        self.node_val = np.array(role['va'])
        self.node_test = np.array(role['te'])

        self.class_arr = class_arr
        self.adj_full = adj_full
        self.adj_train = adj_train
        self.adj_full_rev = self.adj_full.transpose()
        self.adj_train_rev = self.adj_train.transpose()

        assert self.class_arr.shape[0] == self.adj_full.shape[0]

        # below: book-keeping for mini-batch
        self.placeholders = placeholders
        self.node_subgraph = None
        self.batch_num = -1

        self.method_sample = None
        self.sampled_subgraphs_num = 0
        self.subgraphs_remaining = []


    def set_sampler(self,train_phases):
        self.sampled_subgraphs_num = 0
        self.subgraphs_remaining = []
        self.method_sample = train_phases['sampler']
        if self.method_sample == 'frontier':
            self.size_subg_budget = train_phases['size_subgraph']
            self.size_frontier = train_phases['size_frontier']
            self.graph_sampler = frontier_sampling(self.adj_train,self.adj_full,\
                self.node_train,self.size_subg_budget,self.size_frontier)
        else:
            raise NotImplementedError


    def minibatch_train_feed_dict(self, dropout, is_val=False, is_test=False):
        """ DONE """
        if is_val or is_test:
            self.node_subgraph = np.arange(self.class_arr.shape[0])
            _adj_rev = self.adj_full_rev
        else:
            if len(self.subgraphs_remaining) == 0:
                if self.method_sample == 'frontier':
                    _args = {'frontier':None}
                else:
                    _args = dict()
                self.par_graph_sample('train',0, '', self.sampled_subgraphs_num,_args)

            self.node_subgraph = self.subgraphs_remaining.pop()
            self.size_subgraph = len(self.node_subgraph)
            _adj_rev = self.adj_train_rev
            self.batch_num += 1
        self.node_subgraph.sort()
        feed_dict = dict()
        feed_dict.update({self.placeholders['node_subgraph']: self.node_subgraph})
        feed_dict.update({self.placeholders['labels']: self.class_arr[self.node_subgraph]})
        feed_dict.update({self.placeholders['dropout']: dropout})
        
        # obtain adj of subgraph

        rows = []
        cols = []
        arr_bit = np.zeros(_adj_rev.shape[0])-1
        arr_bit[self.node_subgraph] = np.arange(self.node_subgraph.size)
        for iv,v in enumerate(self.node_subgraph):
            _cur_neigh = _adj_rev.indices[_adj_rev.indptr[v]:_adj_rev.indptr[v+1]]
            indicator = arr_bit[_cur_neigh]
            _ = indicator[indicator>-1]
            cols.extend(list(_))
            rows.extend([iv]*len(_))
        adj = sp.csr_matrix(([1]*len(rows),(rows,cols)),shape=(self.node_subgraph.size,self.node_subgraph.size))
        _num_edges = len(adj.nonzero()[1])
        _num_vertices = len(self.node_subgraph)
        print('subgraph: {} vertices, {} edges, {:5.2f} degree\t\tis_val: {}, is_test: {}'\
            .format(_num_vertices,_num_edges,_num_edges/_num_vertices,is_val,is_test))
        _indices_ph = np.column_stack(adj.nonzero())
        _shape_ph = adj.shape
        _diag_shape = (adj.shape[0],adj.shape[0])
        _norm_diag = sp.dia_matrix((1/adj.sum(1).flatten(),0),shape=_diag_shape)
        _adj_norm = _norm_diag.dot(adj)
        feed_dict.update({self.placeholders['adj_subgraph']: \
            tf.SparseTensorValue(_indices_ph,_adj_norm.data,_shape_ph)})
        return feed_dict, self.class_arr[self.node_subgraph]



    def par_graph_sample(self,mode,save,prefix,sampled_subgraph_num,args_dict):
        if self.num_proc > 1:
            printf('par sampling',type='WARN')
            output_subgraphs = mp.Queue()
            processes = [mp.Process(target=self.graph_sampler.sample,args=(output_subgraphs,i,mode,args_dict['frontier'])) for i in range(self.num_proc)]
            for p in processes:
                p.start()
            num_proc_done = 0
            subgraph_assembler = dict()
            for i in range(self.num_proc):
                subgraph_assembler[i] = []
            while True:
                seg = output_subgraphs.get()
                subgraph_assembler[seg[0]].extend(seg[2])
                if seg[1] == 0:
                    num_proc_done += 1
                if num_proc_done == self.num_proc:
                    break
            for p in processes:
                p.join()
            for k,v in subgraph_assembler.items():
                self.subgraphs_remaining.append(np.array(v))
            #self.subgraphs_remaining.extend([output_subgraphs.get() for p in processes])
        else:
            ret = self.graph_sampler.sample(None,0,mode,args_dict['frontier'])
            self.subgraphs_remaining.append(np.array(ret))



    def num_training_batches(self):
        """ DONE """
        return math.ceil(self.node_train.shape[0]/float(self.size_subg_budget))

    def shuffle(self):
        """ DONE """
        self.node_train = np.random.permutation(self.node_train)
        self.batch_num = -1

    def end(self):
        """ DONE """
        return (self.batch_num+1)*self.size_subg_budget >= self.node_train.shape[0]
