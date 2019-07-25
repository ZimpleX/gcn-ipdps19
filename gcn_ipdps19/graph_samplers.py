import numpy as np
import json
import random
import abc
import time
import math
import pdb
from math import ceil


class graph_sampler:
    __metaclass__ = abc.ABCMeta
    PIPESIZE = 8000
    def __init__(self,adj_train,adj_full,node_train,size_subgraph):
        assert adj_train.shape == adj_full.shape
        self.adj_train = adj_train
        self.adj_full = adj_full
        self.node_train = np.unique(node_train)
        self.mask_train = np.array([False]*self.adj_full.shape[0])
        # mask to check if the node is a training node --> for assertion
        self.mask_train[self.node_train] = True
        # size in terms of number of vertices in subgraph
        self.size_subgraph = size_subgraph
        self.coverage_v_train = np.array([float('inf')]*(self.adj_full.shape[0]))
        self.coverage_v_train[self.node_train] = 0.
        self.name_sampler = 'None'
        self.node_subgraph = None
        # =======================
        self.arr_deg_train = np.zeros(self.adj_train.shape[0])
        self.arr_deg_full = np.zeros(self.adj_full.shape[0])
        _1 = np.unique(self.adj_train.nonzero()[0],return_counts=True)
        _2 = np.unique(self.adj_full.nonzero()[0],return_counts=True)
        self.arr_deg_train[_1[0]] = _1[1]
        self.arr_deg_full[_2[0]] = _2[1]
        self.avg_deg_train = self.arr_deg_train.mean()
        self.avg_deg_full = self.arr_deg_full.mean()

        

    @abc.abstractmethod
    def sample(self,output_subgraphs,tid,phase,**kwargs):
        assert phase in ['train','val','test']


class frontier_sampling(graph_sampler):

    def __init__(self,adj_train,adj_full,node_train,size_subgraph,size_frontier):
        super().__init__(adj_train,adj_full,node_train,size_subgraph)
        self.size_frontier = size_frontier
        self.deg_full = np.bincount(self.adj_full.nonzero()[0])
        self.deg_train = np.bincount(self.adj_train.nonzero()[0])
        self.name_sampler = 'FRONTIER'

    def double_arr_indicator(self, arr_indicator,k=2):
        a = np.zeros((k*arr_indicator.size,2),dtype=arr_indicator.dtype)
        a[:arr_indicator.shape[0]] = arr_indicator
        return a


    def sample(self,output_subgraphs,tid,phase,frontier=None):
        m = self.size_frontier
        super().sample(output_subgraphs,tid,phase)
        t1 = time.time()
        assert m is not None or frontier is not None
        EPSILON = 1e-5
        if frontier is None:
            frontier = random.choices(self.node_train,k=m)
            update_coverage_v_train = False#True
        else:
            m = frontier.size
            update_coverage_v_train = False
        _adj = self.adj_train if phase=='train' else self.adj_full
        _deg = self.deg_train if phase=='train' else self.deg_full
        _max_deg = 10000
        lambd = 0
        _arr_deg = self.arr_deg_train if phase=='train' else self.arr_deg_full
        _arr_deg = np.clip(_arr_deg,0,_max_deg) + lambd
        _avg_deg = _arr_deg.mean()
        node_subgraph = []
        # ======================================================
        alpha = 2
        # ---------------------
        # arr_indicator:
        # suppose 2 nodes, u w/ deg 3 and v w/ deg 4
        # arr_indicator is a (7,2) shape array
        #  u  u  u  v  v  v  v
        # -3  1  2 -4  1  2  3
        #  1  1  1  1  2  2  2
        arr_indicator = np.zeros((int(alpha*_avg_deg*m),2),dtype=np.int64)
        deg_cumsum = np.cumsum([0]+[_arr_deg[f] for f in frontier]).astype(np.int64)
        end_idx = deg_cumsum[-1]
        if end_idx > arr_indicator.shape[0]:
            print('doubling')
            arr_indicator = self.double_arr_indicator(arr_indicator,k=ceil(end_idx/arr_indicator.shape[0]))
        for i,mi in enumerate(frontier):
            arr_indicator[deg_cumsum[i]:deg_cumsum[i+1],0] = mi
            arr_indicator[deg_cumsum[i]:deg_cumsum[i+1],1] = np.arange(deg_cumsum[i+1]-deg_cumsum[i])
            arr_indicator[deg_cumsum[i],1] = deg_cumsum[i] - deg_cumsum[i+1]
 

        for cur_size in range(m,self.size_subgraph+1):
            while True:
                idx = random.randint(0,end_idx-1)
                if arr_indicator[idx,0]:
                    break
            selected_v,offset = arr_indicator[idx]
            # *********************
            idx = idx if offset<0 else idx-offset
            idx = int(idx)
            offset = -1*(offset if offset<0 else arr_indicator[idx,1])
            offset = int(offset)
            arr_indicator[idx:idx+offset] = 0
            
            #neighs = _adj.indices[_adj.indptr[selected_v]:_adj.indptr[selected_v+1]]
            num_neighs = _adj.indptr[selected_v+1]-_adj.indptr[selected_v]
            #new_frontier = neighs[random.randint(0,neighs.size-1)]
            new_frontier = _adj.indices[_adj.indptr[selected_v]+random.randint(0,num_neighs-1)]
            node_subgraph.append(new_frontier)
            _deg = min(_adj.indptr[new_frontier+1]-_adj.indptr[new_frontier],_max_deg)+lambd
            if end_idx+_deg > arr_indicator.shape[0]:
                # shift arr_indicator to fill in the gaps of 0
                _start = np.where(arr_indicator[:,1]<0)[0].astype(np.int64)

                _end = _start-arr_indicator[_start,1]
                _end[1:] = _end[:-1]
                _end[0] = 0
                delta = _start-_end
                delta = np.cumsum(delta).astype(np.int64)
                end_idx = (-arr_indicator[_start,1]).sum().astype(np.int64)
                for i,stepi in enumerate(delta):
                    _s = _start[i]
                    _e = _start[i]-arr_indicator[_start[i],1]
                    arr_indicator[_s-stepi:_e-stepi] = arr_indicator[_s:_e]
                if end_idx+_deg > arr_indicator.shape[0]:
                    print('doubling')
                    arr_indicator = self.double_arr_indicator(arr_indicator,k=ceil((end_idx+_deg)/arr_indicator.shape[0]))
            arr_indicator[end_idx:end_idx+_deg,0] = new_frontier
            arr_indicator[end_idx:end_idx+_deg,1] = np.arange(_deg)
            arr_indicator[end_idx,1] = -_deg
            end_idx += _deg

       
        # ======================================================
        node_subgraph.extend(list(frontier))
        node_subgraph = list(set(node_subgraph))
        if update_coverage_v_train:
            self.coverage_v_train[node_subgraph] += 1
        t2 = time.time()
        #print('sampling time: {:.2f}s'.format(t2-t1))
        if output_subgraphs is not None:
            num_seg = int(math.ceil(len(node_subgraph)/self.PIPESIZE))
            for i in reversed(range(num_seg)):
                output_subgraphs.put([tid,i,node_subgraph[i*self.PIPESIZE:(i+1)*self.PIPESIZE]])
        else:
            self.node_subgraph = node_subgraph
            return node_subgraph

