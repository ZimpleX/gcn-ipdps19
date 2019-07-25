import sys
import yaml

import os
import tensorflow as tf
import numpy as np
import time as ttime
import datetime
import pdb


from gcn_ipdps19.inits import *
from gcn_ipdps19.models import GCN_Subgraph
from gcn_ipdps19.minibatch import Minibatch
from gcn_ipdps19.utils import *
from gcn_ipdps19.metric import *


import subprocess
git_rev = subprocess.Popen("git rev-parse --short HEAD", shell=True, stdout=subprocess.PIPE, universal_newlines=True).communicate()[0]
git_branch = subprocess.Popen("git symbolic-ref --short -q HEAD", shell=True, stdout=subprocess.PIPE, universal_newlines=True).communicate()[0]

os.environ["CUDA_DEVICE_ORDER"]=""

# Set random seed
#seed = 123
#np.random.seed(seed)
#tf.set_random_seed(seed)

timestamp = ttime.time()
timestamp = datetime.datetime.fromtimestamp(int(timestamp)).strftime('%Y-%m-%d %H:%M:%S')


flags = tf.app.flags
FLAGS = flags.FLAGS
# Settings
tf.app.flags.DEFINE_boolean('log_device_placement', False, "Whether to log device placement.")
flags.DEFINE_string('data_prefix', '', 'prefix identifying training data. must be specified.')
#logging, saving, validation settings etc.
flags.DEFINE_string('base_log_dir', '.', 'base directory for logging and saving embeddings')
flags.DEFINE_integer('gpu', 1, "which gpu to use.")
flags.DEFINE_integer('eval_train_every', 10, "How often to evaluate training subgraph accuracy.")

flags.DEFINE_string('train_config', '*.yml', "path to the configuration of training (*.yml)")
flags.DEFINE_boolean('mkl',False,'whether to use mkl library')


os.environ["CUDA_VISIBLE_DEVICES"]=str(FLAGS.gpu)


GPU_MEM_FRACTION = 0.8

f_mean = lambda l: sum(l)/len(l)


def evaluate_full_batch(sess,model,minibatch_iter,mode):
    """
    Full batch evaluation
    """
    num_cls = minibatch_iter.class_arr.shape[-1]
    feed_dict, labels = minibatch_iter.minibatch_train_feed_dict(0.,is_val=True,is_test=True)
    preds,loss = sess.run([model.preds, model.loss], feed_dict=feed_dict)
    nodes_for_acc = minibatch_iter.node_val if mode=='val' else minibatch_iter.node_test
    f1_scores = calc_f1(labels[nodes_for_acc],preds[nodes_for_acc],model.sigmoid_loss)
    return loss, f1_scores[0], f1_scores[1]



def construct_placeholders(num_classes):
    placeholders = {
        'labels' : tf.placeholder(tf.float32, shape=(None, num_classes), name='labels'),
        'node_subgraph' : tf.placeholder(tf.int32, shape=(None), name='node_subgraph'),
        'dropout': tf.placeholder_with_default(0., shape=(), name='dropout'),
        'adj_subgraph' : tf.sparse_placeholder(tf.float32,name='adj_subgraph'),
    }
    return placeholders





#########
# TRAIN #
#########
def prepare(train_data,train_params,dims_gcn):
    adj_full,adj_train,feats,class_arr,role = train_data
    num_classes = class_arr.shape[1]
    # pad with dummy zero vector
    # features = np.vstack([features, np.zeros((features.shape[1],))])

    dims = dims_gcn[:-1]
    loss_type = dims_gcn[-1]

    placeholders = construct_placeholders(num_classes)
    minibatch = Minibatch(adj_full, adj_train, role, class_arr, placeholders)
    model = GCN_Subgraph(num_classes, placeholders,
                feats, dims, train_params, loss=loss_type, logging=True)

    # Initialize session
    sess = tf.Session(config=tf.ConfigProto(device_count={"CPU":2},inter_op_parallelism_threads=44,intra_op_parallelism_threads=44))
    ph_misc_stat = {'val_f1_micro': tf.placeholder(tf.float32, shape=()),
                    'val_f1_macro': tf.placeholder(tf.float32, shape=()),
                    'train_f1_micro': tf.placeholder(tf.float32, shape=()),
                    'train_f1_macro': tf.placeholder(tf.float32, shape=()),
                    'size_subgraph': tf.placeholder(tf.int32, shape=())}
    merged = tf.summary.merge_all()

    with tf.name_scope('summary'):
        _misc_val_f1_micro = tf.summary.scalar('val_f1_micro', ph_misc_stat['val_f1_micro'])
        _misc_val_f1_macro = tf.summary.scalar('val_f1_macro', ph_misc_stat['val_f1_macro'])
        _misc_train_f1_micro = tf.summary.scalar('train_f1_micro', ph_misc_stat['train_f1_micro'])
        _misc_train_f1_macro = tf.summary.scalar('train_f1_macro', ph_misc_stat['train_f1_macro'])
        _misc_size_subgraph = tf.summary.scalar('size_subgraph',ph_misc_stat['size_subgraph'])

    misc_stats = tf.summary.merge([_misc_val_f1_micro,_misc_val_f1_macro,_misc_train_f1_micro,_misc_train_f1_macro,_misc_size_subgraph])
    summary_writer = tf.summary.FileWriter(log_dir(train_params,FLAGS.data_prefix,git_branch,git_rev,timestamp), sess.graph)
    # Init variables
    sess.run(tf.global_variables_initializer())
    return model,minibatch, sess, [merged,misc_stats],ph_misc_stat, summary_writer



def train(train_phases,train_params,dims_gcn,model,minibatch,\
            sess,train_stat,ph_misc_stat,summary_writer):
    avg_time = 0.0

    timing_steps = 0

    saver = tf.train.Saver()

    epoch_ph_start = 0
    for ip,phase in enumerate(train_phases):
        minibatch.set_sampler(phase)
        num_batches = minibatch.num_training_batches()
        printf('START PHASE {:4d}'.format(ip),style='underline')
        for e in range(epoch_ph_start,phase['end']):
            printf('Epoch {:4d}'.format(e),style='bold')
            minibatch.shuffle()
            l_loss_tr = list()
            l_f1mic_tr = list()
            l_f1mac_tr = list()
            l_size_subg = list()
            time_epoch = 0
            while not minibatch.end():
                feed_dict, labels = minibatch.minibatch_train_feed_dict(phase['dropout'],is_val=False,is_test=False)
                t0=ttime.time()
                _,__,loss_train,pred_train = sess.run([train_stat[0], \
                        model.opt_op, model.loss, model.preds], feed_dict=feed_dict)
                t1 = ttime.time()
                time_epoch += t1-t0
                if not minibatch.batch_num % FLAGS.eval_train_every:
                    f1_mic,f1_mac = calc_f1(labels,pred_train,dims_gcn[-1])
                    #printf("Iter {:4d}\ttrain loss {:.5f}\tmic {:5f}\tmac {:5f}".format(
                    #    minibatch.batch_num,loss_train,f1_mic,f1_mac))
                    l_loss_tr.append(loss_train)
                    l_f1mic_tr.append(f1_mic)
                    l_f1mac_tr.append(f1_mac)
                    l_size_subg.append(minibatch.size_subgraph)
            loss_val,f1mic_val,f1mac_val = evaluate_full_batch(sess,model,minibatch,mode='val')
            printf('  val loss {:.5f}\tmic {:.5f}\tmac {:.5f}'.format(loss_val,f1mic_val,f1mac_val))
            printf('  avg train loss {:.5f}\tmic {:.5f}\tmac {:.5f}'.format(f_mean(l_loss_tr),f_mean(l_f1mic_tr),f_mean(l_f1mac_tr)))
                
            misc_stat = sess.run([train_stat[1]],feed_dict={\
                                    ph_misc_stat['val_f1_micro']: f1mic_val,
                                    ph_misc_stat['val_f1_macro']: f1mac_val,
                                    ph_misc_stat['train_f1_micro']: f_mean(l_f1mic_tr),
                                    ph_misc_stat['train_f1_macro']: f_mean(l_f1mac_tr),
                                    ph_misc_stat['size_subgraph']: f_mean(l_size_subg)})
            # tensorboard visualization
            summary_writer.add_summary(_, e)
            summary_writer.add_summary(misc_stat[0], e)
        epoch_ph_start = phase['end']
    saver.save(sess, 'models/{data}'.format(data=FLAGS.data_prefix.split('/')[-1]),global_step=e)
    save_model_weights(sess, model, FLAGS.data_prefix.split('/')[-1],e,FLAGS.train_config)
    printf("Optimization Finished!",style='red')
    loss, f1_mic, f1_mac = evaluate_full_batch(sess,model,minibatch,mode='val')
    printf("Full validation stats: loss={:.5f}, f1_micro={:.5f}, f1_macro={:.5f}".format(loss,f1_mic,f1_mac),style='yellow')
    loss, f1_mic, f1_mac = evaluate_full_batch(sess,model,minibatch,mode='test')
    printf("Full test stats: loss={:.5f}, f1_micro={:.5f}, f1_macro={:.5f}".format(loss,f1_mic,f1_mac),style='yellow')
 

########
# MAIN #
########

def main(argv=None):
    train_params,train_phases,train_data,dims_gcn = parse_n_prepare(FLAGS)
    model,minibatch,sess,train_stat,ph_misc_stat,summary_writer = \
            prepare(train_data,train_params,dims_gcn)
    train(train_phases,train_params,dims_gcn,model,minibatch,sess,train_stat,ph_misc_stat,summary_writer)


if __name__ == '__main__':
    tf.app.run()
