# import python modules
import tensorflow as tf
import networkx as nx
import pickle as pkl
import numpy as np
import argparse
import os
import scipy

import sys
sys.path.append('/data3/home/mbito/project_dynamic_graphs/tensorflow/models')

# import custom modules
from gcn import *
from gin import *
from sgc import *
from gat import *
from gcn2 import *
from gsage import *
from faconv import *

from generator import generator

if __name__ == '__main__': 

    parser = argparse.ArgumentParser()
    parser.add_argument('--lr', type=float)
    parser.add_argument('--hidden', type=int)
    parser.add_argument('--gpu', type=int)
    parser.add_argument('--idx', type=int)
    parser.add_argument('--mtype', type=str)
    parser.add_argument('--data_path', type=str)
    parser.add_argument('--save_path', type=str)
    
    args = parser.parse_args()
    lr = args.lr
    hidden = args.hidden
    gpu = args.gpu
    idx = args.idx
    mtype = args.mtype
    data_path = args.data_path
    save_path = args.save_path
    
    physical_devices = tf.config.list_physical_devices('GPU')
    tf.config.set_visible_devices(physical_devices[gpu], 'GPU') # mld4 devices range from 0 - 7
    
    train_gen = generator(split='train', data_path=data_path)
    train_dset = train_gen.tf_generator(train_gen.data_generator)

    valid_gen = generator(split='valid', data_path=data_path)
    valid_dset = valid_gen.tf_generator(valid_gen.data_generator)

    test_gen = generator(split='test', data_path=data_path)
    test_dset = test_gen.tf_generator(test_gen.data_generator)
    
    optimizer = tf.keras.optimizers.Adam(learning_rate=lr)
    loss = tf.keras.losses.BinaryCrossentropy()
    
    if mtype == 'gcn':
        model = GCN(num_layers=hidden, n_classes=1, bias=True, dropout=0, g_dropout=0, optimizer=optimizer, loss=loss)
    elif mtype == 'sgc':
        model = SGC(num_layers=hidden, n_classes=1, bias=True, dropout=0, g_dropout=0, optimizer=optimizer, loss=loss)
    elif mtype == 'gin':
        model = GIN(num_layers=hidden, n_classes=1, bias=True, dropout=0, g_dropout=0, optimizer=optimizer, loss=loss)
    elif mtype == 'gat':
        model = GAT(num_layers=hidden, n_classes=1, bias=True, dropout=0, optimizer=optimizer, loss=loss)
    elif mtype == 'gsage':
        model = GSAGE(num_layers=hidden, n_classes=1, bias=True, dropout=0, optimizer=optimizer, loss=loss)
    elif mtype == 'gcnii':
        model = GCNII(num_layers=hidden, n_classes=1, bias=True, dropout=0, optimizer=optimizer, loss=loss)
    elif mtype == 'fagcn':
        model = FAGCN(num_layers=hidden, n_classes=1, bias=True, dropout=0, optimizer=optimizer, loss=loss)
    else:
        raise NotImplementedError()

    model.train(train_dset, valid_dset, epochs=100, early_stopping=5)

    test_auc, test_err, aucs, val_auc, preds = model.evaluate(valid_dset, test_dset)
    
    performance = {'test_auc': test_auc, 'test_err': test_err, 'val_auc': val_auc, 'aucs': aucs}
    
    if not os.path.exists(save_path): 
        os.mkdir(save_path)
    
    with open(os.path.join(save_path, f'{mtype}_results_{idx}.pkl'), 'wb') as file: 
        pkl.dump((performance, args), file)
        
