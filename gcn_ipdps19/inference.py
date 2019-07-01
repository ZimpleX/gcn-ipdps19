import pickle
import tensorflow as tf
from gcn_ipdps19.utils import *
from gcn_ipdps19.supervised_train import evaluate_full_batch,construct_placeholders,FLAGS
from gcn_ipdps19.minibatch import NodeMinibatchIterator
from gcn_ipdps19.supervised_models import SupervisedGraphsage


# flags to run:
#       --data_prefix   <./data/ppi>
#       --model         <./model/*.pkl>
#       --train_config  <./train_config/*.yaml>



def load_pretrained_model(name):
    with open(name,'rb') as f:
        weights = pickle.load(f)
    return {'dense': [weights['dense_weight'],weights['dense_bias']],
            'meanaggr': [[weights['neigh_weight_0'],weights['self_weight_0']],
                         [weights['neigh_weight_1'],weights['self_weight_1']]]}

def main(argv=None):
    train_params,train_phases,train_data,dims_gcn = parse_n_prepare(FLAGS)
    adj_full,adj_train,feats,class_arr,role = train_data
    num_classes = class_arr.shape[1]

    dims = dims_gcn[:-1]
    loss_type = dims_gcn[-1]

    placeholders = construct_placeholders(num_classes)
    minibatch = NodeMinibatchIterator(adj_full, adj_train, role, class_arr, placeholders)
    sess = tf.Session(config=tf.ConfigProto(device_count={"CPU":2},inter_op_parallelism_threads=44,intra_op_parallelism_threads=44))

    pretrained_mpdel = load_pretrained_model(FLAGS.model)
    model = SupervisedGraphsage(num_classes,placeholders,feats,dims,train_params,\
                loss=loss_type,model_pretrain=pretrained_mpdel,logging=True)

    sess.run(tf.global_variables_initializer())

    loss, f1_mic, f1_mac, duration = evaluate_full_batch(sess,model,minibatch)
    print("Full validation stats:",\
            "loss=", "{:.5f}".format(loss),\
            "f1_micro=", "{:.5f}".format(f1_mic),\
            "f1_macro=", "{:.5f}".format(f1_mac))
    loss, f1_mic, f1_mac, duration = evaluate_full_batch(sess,model,minibatch,is_val=False)
    print("Full test stats:",\
            "loss=", "{:.5f}".format(loss),\
            "f1_micro=", "{:.5f}".format(f1_mic),\
            "f1_macro=", "{:.5f}".format(f1_mac))


if __name__ == '__main__':
    tf.app.run()
