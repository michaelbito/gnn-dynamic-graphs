# import python modules
import tensorflow as tf
import networkx as nx
import pickle as pkl
import numpy as np
import argparse
import shutil
import sys
import os

# import custom modules
sys.path.append('/data3/home/mbito/project_dynamic_graphs/tensorflow/evaluation')
from sis_static import SIStatic
from sis_dynamic import SIDynamic

def generate_dataset(g_train, g_valid, g_test, exp_name='sis', data_path='/data1/mbito/dynamic_graphs_tensorflow/'):
    
    data_path = os.path.join(data_path, exp_name)
    if not os.path.exists(data_path):
        os.mkdir(data_path)
    
    if os.path.exists(data_path):
        shutil.rmtree(data_path)
    os.mkdir(data_path)

    train_path, valid_path, test_path = os.path.join(data_path, 'train'), os.path.join(data_path, 'valid'), os.path.join(data_path, 'test')
    os.mkdir(train_path), os.mkdir(valid_path), os.mkdir(test_path)
    
    with open(os.path.join(data_path, f'history.pkl'), 'wb') as file: 
        pkl.dump(g_test, file)
    
    n_train, n_valid = 0, 0
    for i in range(len(g_train)): 
        X, y, y_mask, adj, adj_mean, adj_norm = process_graph_features(g_train[i])
        if np.unique(y[y_mask]).shape[0] > 1: 
            with open(os.path.join(train_path, f'g_{n_train}.pkl'), 'wb') as file: 
                pkl.dump((X, y, y_mask, adj, adj_mean, adj_norm), file)
            n_train+=1
            
    for i in range(len(g_valid)): 
        X, y, y_mask, adj, adj_mean, adj_norm = process_graph_features(g_valid[i])
        if np.unique(y[y_mask]).shape[0] > 1: 
            with open(os.path.join(valid_path, f'g_{n_valid}.pkl'), 'wb') as file: 
                pkl.dump((X, y, y_mask, adj, adj_mean, adj_norm), file)
            n_valid+=1
            
    for i in range(len(g_test)): 
        X, y, y_mask, adj, adj_mean, adj_norm = process_graph_features(g_test[i])
        with open(os.path.join(test_path, f'g_{i}.pkl'), 'wb') as file: 
            pkl.dump((X, y, y_mask, adj, adj_mean, adj_norm), file)

def process_graph_features(graph): 
    adj = get_adj(nx.adjacency_matrix(graph).todense())
    adj_mean = get_adj_mean(nx.adjacency_matrix(graph).todense())
    adj_norm = get_adj_normalize(nx.adjacency_matrix(graph).todense())
    X = np.array(list(nx.get_node_attributes(graph, 'features').values()))
    y = np.array(list(nx.get_node_attributes(graph, 'exposure').values()))
    y_mask = (np.array(list(nx.get_node_attributes(graph, 'positive').values())) == 0)
    
    return X, y, y_mask, adj, adj_mean, adj_norm

def get_adj(adj, add_self_connections=True, return_dense=True): 
    '''
    return non-normalized adjacency matrix in tensorflow
    '''
    adj = tf.sparse.add(tf.cast(tf.sparse.from_dense(adj), dtype=tf.dtypes.double), tf.sparse.eye(adj.shape[0], dtype=tf.dtypes.double))
    if return_dense: 
        return tf.cast(tf.sparse.to_dense(adj), dtype=tf.dtypes.float32)
    else: 
        return tf.sparse.from_dense(tf.cast(adj, dtype=tf.dtypes.float32))
    
def get_adj_mean(adj, add_self_connections=True, return_dense=True): 
    '''
    Rowise normalize adjacency matrix according to D^-1 A in tensorflow
    '''
    d = np.sum(np.array(adj), axis=1)
    d[d==0] = 1 # avoid nans by replacing 0^-1 with 1^-1
    d = np.diag(np.float_power(d, -1))
    d = tf.convert_to_tensor(d)
    d_matrix = tf.sparse.from_dense(d) # D^-1
    
    adj = tf.cast(tf.sparse.from_dense(adj), dtype=tf.dtypes.double)
    adj_norm = tf.linalg.matmul(tf.sparse.to_dense(d_matrix), tf.sparse.to_dense(adj), a_is_sparse=True, b_is_sparse=True)
    
    if return_dense: 
        return tf.cast(adj_norm, dtype=tf.dtypes.float32)
    else: 
        return tf.sparse.from_dense(tf.cast(adj_norm, dtype=tf.dtypes.float32))

def get_adj_normalize(adj, add_self_connections=True, return_dense=True): 
    '''
    Normalize adjacency matrix according to D^-1/2 A D^-1/2 with sparse matrix operations in tensorflow
    '''
    adj = tf.sparse.add(tf.cast(tf.sparse.from_dense(adj), dtype=tf.dtypes.double), tf.sparse.eye(adj.shape[0], dtype=tf.dtypes.double))
    
    d = tf.sparse.reduce_sum(adj, 1)
    d = tf.math.pow(d, tf.constant(-0.5, shape=(d.shape[0], ), dtype=tf.dtypes.double))
    d_matrix = tf.sparse.from_dense(tf.linalg.diag(d))
    
    norm_a = tf.linalg.matmul(tf.sparse.to_dense(d_matrix), tf.sparse.to_dense(adj), a_is_sparse=True, b_is_sparse=True)
    norm_a = tf.linalg.matmul(norm_a, tf.sparse.to_dense(d_matrix), a_is_sparse=True, b_is_sparse=True)
    
    if return_dense: 
        return tf.cast(norm_a, dtype=tf.dtypes.float32)
    else: 
        return tf.sparse.from_dense(tf.cast(norm_a, dtype=tf.dtypes.float32))
    
def create_dynamic_graph(edges, window_size): 
    '''
    returns a dynamic graph
    inputs: 
        edges (|E| x 3 matrix) : each row represents a temporal edge (u, v, t)
        widnow_size (int) : the time range of each graph in the dynamic graph
    
    outputs: 
        graphs (list of static graphs)
    '''
    
    def process_edges(edges): 
        '''first sort edges by time, then remap edge times to integer increments ie. [0, total_timesteps]'''
        # sort edges by time
        edges = edges[np.argsort(edges[:, 2]), :]
        
        # remap edge times to integer increments
        edges_remapped = np.zeros(edges.shape)
        t_original = edges[0, 2]
        t_remapped = 0
        for t in range(edges.shape[0]): 
            if edges[t, 2] > t_original: 
                t_original = edges[t, 2]
                t_remapped += 1

            edges_remapped[t, :] = np.array([edges[t, 0], edges[t, 1], t_remapped])

        return edges_remapped
    
    edges = process_edges(edges)
    graphs = []
    edge_times = edges[:, 2]
    total_timesteps = np.unique(edge_times).shape[0] // window_size # floor divide ignores the remainder timesteps
    
    # iterate through temporal edges and create total_timesteps static graphs
    for t in range(total_timesteps): 
        # create a mask that selects edges and nodes within the range [t * window_size, (t+1) * window_size]
        t_start = t * window_size
        t_end = (t + 1) * window_size
        mask_start = (edge_times >= t_start).astype(int)
        mask_end = (edge_times < t_end).astype(int)
        mask = (mask_start + mask_end) == 2
        edges_t = edges[mask, :2]
        
        # create a static graph using all the edges and nodes within the range [t * window_size, (t+1) * window_size]
        g = nx.Graph()
        g.add_edges_from(edges_t)
        graphs.append(g)
    
    return graphs

def generate_graph(graph_name, graph_param): 
    ''' generate initial graph '''
    if graph_name == 'regular': 
        graph = nx.random_regular_graph(int(graph_param), 1000)
    elif graph_name == 'powerlaw': 
        graph = nx.barabasi_albert_graph(1000, int(graph_param))
    elif graph_name == 'random':
        graph = nx.binomial_graph(1000, graph_param)
    elif graph_name == 'block':
        # hardcode the block model parameters
        N_BLOCKS = 50
        BLOCK_SIZE = 20
        P_IN = .5
        P_OUT = .001
        
        graph = nx.random_partition_graph([BLOCK_SIZE for i in range(N_BLOCKS)], P_IN, P_OUT)
    else: 
        raise NotImplementedError()
        
    return graph

if __name__ == '__main__': 
    parser = argparse.ArgumentParser()
    parser.add_argument('--param', type=float)
    parser.add_argument('--window_size', type=int)
    parser.add_argument('--runs', type=int, default=20)
    parser.add_argument('--timesteps', type=int, default=30)
    parser.add_argument('--inf_params', type=float, nargs=2)
    parser.add_argument('--sus_params', type=float, nargs=2)
    parser.add_argument('--gpu', type=int, default=0)
    parser.add_argument('--real', type=int, default=-1)
    parser.add_argument('--graph_name', type=str)
    
    args = parser.parse_args()
    param = args.param
    runs = args.runs
    timesteps = args.timesteps
    inf_alpha, inf_beta = args.inf_params
    sus_alpha, sus_beta = args.sus_params
    gpu = args.gpu
    real = args.real
    window_size = args.window_size
    graph_name = args.graph_name
    
    # set the device
    tf.config.set_visible_devices(tf.config.list_physical_devices('GPU')[gpu], 'GPU') 
    
    if real == -1: 
        g_train = []
        g_valid = []
        g_test = []

        for r in range(runs): 
            g_tr = generate_graph(graph_name, param)
            g_va = generate_graph(graph_name, param)
            g_te = generate_graph(graph_name, param)

            g_train+=SIStatic(g=g_tr.copy(), timesteps=timesteps, inf_alpha=inf_alpha, inf_beta=inf_beta, sus_alpha=sus_alpha, sus_beta=sus_beta).graphs
            g_valid+=SIStatic(g=g_va.copy(), timesteps=timesteps, inf_alpha=inf_alpha, inf_beta=inf_beta, sus_alpha=sus_alpha, sus_beta=sus_beta).graphs
            g_test+=SIStatic(g=g_te.copy(), timesteps=timesteps, inf_alpha=inf_alpha, inf_beta=inf_beta, sus_alpha=sus_alpha, sus_beta=sus_beta).graphs

        generate_dataset(g_train, g_valid, g_test, exp_name=f'si_{graph_name}_{runs}r_{timesteps}t')
    else:
        dataset_params = {0: {'path': 'CollegeMsg.txt', 'window_size': 2000}, 
                          1: {'path': 'email-Eu-core-temporal.txt', 'window_size': 5000},
                          2: {'path': 'sx-mathoverflow.txt', 'window_size': 12000}, 
                          3: {'path': 'soc-sign-bitcoinalpha.csv', 'window_size': 100},
                          4: {'path': 'soc-sign-bitcoinotc.csv', 'window_size': 1000},
                          5: {'path': 'soc-redditHyperlinks-body.tsv', 'window_size': 10000},
                          6: {'path': 'soc-redditHyperlinks-title.tsv', 'window_size': 10000}}

        if real in [0, 1, 2]:
            edges = np.loadtxt(f'/data1/mbito/dynamic_graphs_social/{dataset_params[real]["path"]}')
        elif real in [3, 4]:
            edges = np.genfromtxt(f'/data1/mbito/dynamic_graphs_social/{dataset_params[real]["path"]}', delimiter=',')[:, [0, 1, 3]]
        elif real in [5, 6]:
            edges = pd.read_csv(f'/data1/mbito/dynamic_graphs_social/{dataset_params[real]["path"]}', delimiter='\t')
            raise NotImplementedError()

        graphs = create_dynamic_graph(edges, window_size=window_size)
        g_train = []
        g_valid = []
        g_test = []

        for r in range(runs): 
            g_train += SIDynamic(graphs, inf_alpha=inf_alpha, inf_beta=inf_beta, sus_alpha=sus_alpha, sus_beta=sus_beta).graphs[:20]
            g_valid += SIDynamic(graphs, inf_alpha=inf_alpha, inf_beta=inf_beta, sus_alpha=sus_alpha, sus_beta=sus_beta).graphs[:20]
            g_test += SIDynamic(graphs, inf_alpha=inf_alpha, inf_beta=inf_beta, sus_alpha=sus_alpha, sus_beta=sus_beta).graphs[:20]

        generate_dataset(g_train, g_valid, g_test, exp_name=f'si_{graph_name}_{runs}r_{len(graphs)}t')