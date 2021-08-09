# FAAN
Source code for EMNLP2020 paper: [*Adaptive Attentional Network for Few-Shot Knowledge Graph Completion*](https://aclanthology.org/2020.emnlp-main.131/).

Few-shot Knowledge Graph (KG) completion is a focus of current research, where each task aims at querying unseen facts of a relation given few-shot reference entity pairs. 
This work proposes an adaptive attentional network for few-shot KG completion by learning adaptive entity and reference representations. Evaluation in link prediction on two public datasets shows that our approach achieves new state-of-the-art results with different few-shot sizes.

# Requirements

```
python 3.6
Pytorch == 1.1.0
CUDA: 9.0
GPU: Tesla T4
```

# Datasets

We adopt Nell and Wiki datasets to evaluate our model, FAAN.
The orginal datasets and pretrain embeddings are provided from [xiong's repo](https://github.com/xwhan/One-shot-Relational-Learning). 
For convenience, the datasets can be downloaded from [Nell data](https://sites.cs.ucsb.edu/~xwhan/datasets/nell.tar.gz)
and [Wiki data](https://sites.cs.ucsb.edu/~xwhan/datasets/wiki.tar.gz). 
The pre-trained embeddings can be downloaded from [Nell embeddings](https://drive.google.com/file/d/1XXvYpTSTyCnN-PBdUkWBXwXBI99Chbps/view?usp=sharing)
 and [Wiki embeddings](https://drive.google.com/file/d/1_3HBJde2KVMhBgJeGN1-wyvW88gRU1iL/view?usp=sharing).
Note that all these files were provided by xiong and we just select what we need here. 
All the dataset files and the pre-trained TransE embeddings should be put into the directory ./NELL and ./Wiki, respectively.

# How to run
To achieve the best performance, pls train the models as follows:

#### Nell

```
python trainer.py --weight_decay 0.0 --prefix nell.5shot
```

#### Wiki

```
python trainer.py --dataset wiki --embed_dim 50 --num_transformer_layers 4 --num_transformer_heads 8 --dropout_input 0.3 --dropout_layers 0.2 --lr 6e-5 --prefix wiki.5shot
```

To test the trained models, pls run as follows:

#### Nell

```
python trainer.py --weight_decay 0.0 --prefix nell.5shot_best --test
```

#### Wiki

```
python trainer.py --dataset wiki --embed_dim 50 --num_transformer_layers 4 --num_transformer_heads 8 --dropout_input 0.3 --dropout_layers 0.2 --lr 6e-5 --prefix wiki.5shot --test
```

# Citation

If you find this code useful, pls cite our work:

```
@inproceedings{Sheng2020:FAAN,
  author    = {Jiawei Sheng and
               Shu Guo and
               Zhenyu Chen and
               Juwei Yue and
               Lihong Wang and
               Tingwen Liu and
               Hongbo Xu},
  title     = {Adaptive Attentional Network for Few-Shot Knowledge Graph Completion},
  booktitle = {Proceedings of the 2020 Conference on Empirical Methods in Natural
               Language Processing, {EMNLP} 2020, Online, November 16-20, 2020},
  pages     = {1681--1691},
  publisher = {Association for Computational Linguistics},
  year      = {2020}
}
```
