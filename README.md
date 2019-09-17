# Accurate, Efficient and Scalable Graph Embedding

Open source code for the GCN training algorithm in the IEEE/IPDPS'19 paper.

Right now, for ease of extending the framework, we have integrated our sampling and feature propagation algorithm into Tensorflow (where the subgraph feature propagation as a self-defined operation).

### Contact

Hanqing Zeng: zengh@usc.edu,
Hongkuan Zhou: hongkuaz@usc.edu

## Dependencies

* python >= 3.6.8
* tensorflow >=1.12.0
* numpy >= 1.14.3
* scipy >= 1.1.0
* scikit-learn >= 0.19.1
* openmp >= 4.0
* mkl >= 2018.0.2

## Dataset

There are three datasets (PPI, Reddit and Yelp) available via this [Google Drive link](https://drive.google.com/open?id=1zycmmDES39zVlbVCYs88JTJ1Wm5FbfLz). Rename the folder to `data` at the root directory.  The directory structure should be as below:

```
GCN-IPDPS19/
│   README.md
│   run_gs_subgraph.sh
│   ... 
│
└───gcn_ipdps19/
│   │   models.py
│   │   train.py
│   │   ...
│   
└───data/
│   └───ppi/
│   │   │    adj_train.npz
│   │   │    adj_full.npz
│   │   │    ...
│   │   
│   └───reddit/
│   │   │    ...
│   │
│   └───...
│
```

## Run Training

There are two options to run the code on CPU with mkl. You can either use the `--mkl` flag to directly call C++ mkl libraries or use the mkl-built tensorflow.

To run the code on CPU with mkl-built tensorflow:

`./run_training.sh <dataset_name> <path to train_config yml>`

To run the code on CPU with python-build tensorflow:

`./run_training.sh <dataset_name> <path to train_config yml> --mkl`

To run the code on GPU:

`./run_training.sh <dataset_name> <path to train_config yml> --gpu <GPU number>`

For example `--gpu 0` will run on the fisrt GPU.


## Citation

```
@article{DBLP:journals/corr/abs-1810-11899,
  author    = {Hanqing Zeng and
               Hongkuan Zhou and
               Ajitesh Srivastava and
               Rajgopal Kannan and
               Viktor K. Prasanna},
  title     = {Accurate, Efficient and Scalable Graph Embedding},
  journal   = {CoRR},
  volume    = {abs/1810.11899},
  year      = {2018},
  url       = {http://arxiv.org/abs/1810.11899},
  archivePrefix = {arXiv},
  eprint    = {1810.11899},
  timestamp = {Wed, 31 Oct 2018 14:24:29 +0100},
  biburl    = {https://dblp.org/rec/bib/journals/corr/abs-1810-11899},
  bibsource = {dblp computer science bibliography, https://dblp.org}
}
```
