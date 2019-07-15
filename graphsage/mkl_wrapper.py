import scipy.linalg.blas as blas
import numpy as np
import os
import tensorflow as tf
from tensorflow.python.framework import ops
from ctypes import *
mkl = cdll.LoadLibrary("libmkl_rt.so")
# num_cpu=40
# mkl.mkl_set_num_threads(byref(c_int(num_cpu)))

# see https://gist.github.com/harpone/3453185b41d8d985356cbe5e57d67342
def py_func(func,inp,Tout,stateful=True,name=None,grad=None):
    rnd_name='PyFuncGrad' + str(np.random.randint(0, 1E+8))
    tf.RegisterGradient(rnd_name)(grad)
    g = tf.get_default_graph()
    with g.gradient_override_map({"PyFunc": rnd_name}):
        return tf.py_func(func, inp, Tout, stateful=stateful, name=name)

def sgemm(a,b,alpha=1.0):
    c=blas.sgemm(alpha,a,b)
    return c

def tf_sgemm(a,b):
    c=py_func(sgemm,[a,b],[tf.float32],grad=_tf_sgemm_grad)
    return c[0]

def _tf_sgemm_grad(op,grad):
    a=op.inputs[0]
    b=op.inputs[1]
    # import pdb; pdb.set_trace()
    # partial gradients with respect to a and b
    return tf_sgemm(grad,tf.transpose(b)),tf_sgemm(tf.transpose(a),grad)

# see https://software.intel.com/en-us/articles/using-intel-mkl-in-your-python-programs
# for multithreading issue, see https://software.intel.com/en-us/articles/recommended-settings-for-calling-intel-mkl-routines-from-multi-threaded-applications
# a_index: int32 array of size (nnz,2)
# a_data: float32 array of size (nnz,)
# a_shape: int32 array of size (2,)
def scoomm(a_index,a_data,a_shape,b,alpha=1.0):
    transa='N'
    m=a_shape[0]
    n=b.shape[1]
    k=a_shape[1]
    alpha=1.0
    beta=0.0
    matdescra='GLNC'
    nnz=a_data.shape[0]
    ldb=b.shape[1]
    c=np.zeros((m,b.shape[1]),dtype=np.float32)
    ldc=c.shape[1]
    c_int_p=POINTER(c_int)
    c_float_p=POINTER(c_float)
    # void mkl_scoomm (const char *transa , const MKL_INT *m , const MKL_INT *n , const MKL_INT *k , const float *alpha , const char *matdescra , const float *val , const MKL_INT *rowind , const MKL_INT *colind , const MKL_INT *nnz , const float *b , const MKL_INT *ldb , const float *beta , float *c , const MKL_INT *ldc );
    mkl.mkl_scoomm(c_char_p(transa.encode('utf-8')),pointer(c_int(m)),pointer(c_int(n)),pointer(c_int(k)),pointer(c_float(alpha)),c_char_p(matdescra.encode('utf-8')),a_data.ctypes.data_as(c_float_p),a_index[0].ctypes.data_as(c_int_p),a_index[1].ctypes.data_as(c_int_p),pointer(c_int(nnz)),b.ctypes.data_as(c_float_p),pointer(c_int(ldb)),pointer(c_float(beta)),c.ctypes.data_as(c_float_p),pointer(c_int(ldc)))
    return c

def tf_scoomm(a_sp,b):
    a_index=tf.dtypes.cast(tf.transpose(a_sp.indices),dtype=tf.int32)
    a_data=a_sp.values
    a_shape=tf.dtypes.cast(a_sp.dense_shape,dtype=tf.int32)
    c=py_func(scoomm,[a_index,a_data,a_shape,b],[tf.float32],grad=_tf_scoomm_grad)
    return c[0]

def _tf_scoomm_grad(op,grad):
    a_index=op.inputs[0]
    a_data=op.inputs[1]
    a_shape=op.inputs[2]
    b=op.inputs[3]
    #####################
    #------WARNING------#
    #####################
    # wrong gradient for the sparse matrix
    # this is not needed in GraphSAINT
    _d_a_index=tf.zeros_like(a_index)
    _d_a_data=tf.zeros_like(a_data)
    _d_a_shape=tf.zeros_like(a_shape)
    a_sp=tf.SparseTensor(indices=tf.dtypes.cast(tf.transpose(a_index),dtype=tf.int64),values=a_data,dense_shape=tf.dtypes.cast(a_shape,dtype=tf.int64))
    _d_b=tf_scoomm(tf.sparse.transpose(a_sp),grad)
    return _d_a_index,_d_a_data,_d_a_shape,_d_b


if __name__=="__main__":
    print("testing tf wrapped mkl functions...")
    os.environ['CUDA_VISIBLE_DEVICES']=''
    sess=tf.Session()
    print("testing sgemm with grad...")
    a=tf.constant([[1,2,3],[2,3,4]],dtype=tf.float32)
    b=tf.constant([[0],[1],[2]],dtype=tf.float32)
    c=tf_sgemm(a,b)
    with sess.as_default():
        print("c=",c.eval())
        print("c should be [[8],[11]]")
        print("dc/da=",tf.gradients(c,a)[0].eval())
        print("dc/da should be [[0,1,2],[0,1,2]]")
        print("dc/db=",tf.gradients(c,b)[0].eval())
        print("dc/db should be [[3],[5],[7]]")
    
    print("testing scoomm with grad...")
    a_sp=tf.SparseTensor(indices=tf.constant([[0,1],[1,1],[1,2]],dtype=tf.int64),values=tf.constant([2,3,4],dtype=tf.float32),dense_shape=tf.constant([2,3],dtype=tf.int64))
    d=tf_scoomm(a_sp,b)
    with sess.as_default():
        print("d=",d.eval())
        print("d should be [[2],[11]]")
        print("dd/db=",tf.gradients(d,b)[0].eval())
        print("dd/db should be [[0],[5],[4]]")

# import pdb; pdb.set_trace()
