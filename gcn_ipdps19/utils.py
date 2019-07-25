import numpy as np
import json
import pdb
import scipy.sparse
from sklearn.preprocessing import StandardScaler
import os
import yaml


def load_data(prefix, normalize=True):
    adj_full = scipy.sparse.load_npz('./{}/adj_full.npz'.format(prefix))
    adj_train = scipy.sparse.load_npz('./{}/adj_train.npz'.format(prefix))
    role = json.load(open('./{}/role.json'.format(prefix)))
    feats = np.load('./{}/feats.npy'.format(prefix))
    class_map = json.load(open('./{}/class_map.json'.format(prefix)))
    class_map = {int(k):v for k,v in class_map.items()}
    assert len(class_map) == feats.shape[0]
    # ---- normalize feats ----
    train_nodes = np.array(list(set(adj_train.nonzero()[0])))
    train_feats = feats[train_nodes]
    scaler = StandardScaler()
    scaler.fit(train_feats)
    feats = scaler.transform(feats)
    # -------------------------
    return adj_full, adj_train, feats, class_map, role


def process_graph_data(adj_full, adj_train, feats, class_map, role):
    """
    setup vertex property map for output classes, train/val/test masks, and feats
    INPUT:
        G           graph-tool graph, full graph including training,val,testing
        feats       ndarray of shape |V|xf
        class_map   dictionary {vertex_id: class_id}
        val_nodes   index of validation nodes
        test_nodes  index of testing nodes
    OUTPUT:
        G           graph-tool graph unchanged
        role        array of size |V|, indicating 'train'/'val'/'test'
        class_arr   array of |V|x|C|, converted by class_map
        feats       array of features unchanged
    """
    num_vertices = adj_full.shape[0]
    if isinstance(list(class_map.values())[0],list):
        num_classes = len(list(class_map.values())[0])
        class_arr = np.zeros((num_vertices, num_classes))
        for k,v in class_map.items():
            class_arr[k] = v
    else:
        num_classes = max(class_map.values()) - min(class_map.values()) + 1
        class_arr = np.zeros((num_vertices, num_classes))
        offset = min(class_map.values())
        for k,v in class_map.items():
            class_arr[k][v-offset] = 1
    return adj_full, adj_train, feats, class_arr, role




def save_model_weights(sess, model, data_name, epoch, train_config):
    import pickle
    key_neigh_weight = 'neigh_weight_{}'
    key_self_weight = 'self_weight_{}'
    key_dense_weight = 'dense_weight'
    key_dense_bias = 'dense_bias'
    out_weights = {key_neigh_weight.format(0): sess.run(model.aggregators[0].vars['neigh_weights']),
                   key_self_weight.format(0): sess.run(model.aggregators[0].vars['self_weights']),
                   key_neigh_weight.format(1): sess.run(model.aggregators[1].vars['neigh_weights']),
                   key_self_weight.format(1): sess.run(model.aggregators[1].vars['self_weights']),
                   key_dense_weight: sess.run(model.node_pred.vars['weights']),
                   key_dense_bias: sess.run(model.node_pred.vars['bias'])}
    outf = "./models/{data}-{method}-{epoch}.pkl"
    with open(outf.format(data=data_name,method=train_config.split('/')[-1].split('.')[-2],epoch=epoch),'wb') as output:
        pickle.dump(out_weights,output)


def parse_n_prepare(flags):
    with open(flags.train_config) as f_train_config:
        train_config = yaml.load(f_train_config)
    dims_gcn = train_config['network']
    train_phases = train_config['phases']
    train_params = train_config['params'][0]
    for ph in train_phases:
        assert 'end' in ph
        assert 'dropout' in ph
        assert 'sampler' in ph
    print("Loading training data..")
    temp_data = load_data(flags.data_prefix)
    train_data = process_graph_data(*temp_data)
    print("Done loading training data..")
    return train_params,train_phases,train_data,dims_gcn





def log_dir(train_params,prefix,git_branch,git_rev,timestamp):
    log_dir = "./raid/user/hzeng/tf_log/" + prefix.split("/")[-1]
    log_dir += "/{ts}-lr_{lr:0.4f}-{gitbranch:s}_{gitrev:s}/".format(
            lr=train_params['lr'],
            gitbranch=git_branch.strip(),
            gitrev=git_rev.strip(),
            ts=timestamp)
    if not os.path.exists(log_dir):
        os.makedirs(log_dir)
    return log_dir



# ------- printing function -------


def printf(msg,style=''):
    _bcolors = {'header': '\033[95m',
                'blue': '\033[94m',
                'green': '\033[92m',
                'yellow': '\033[93m',
                'red': '\033[91m',
                'bold': '\033[1m',
                'underline': '\033[4m'}
    if not style or style == 'black':
        print(msg)
    else:
        print("{color1}{msg}{color2}".format(color1=_bcolors[style],msg=msg,color2='\033[0m'))



