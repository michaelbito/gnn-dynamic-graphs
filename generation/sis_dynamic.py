import sys
import networkx as nx
import numpy as np
import matplotlib.pyplot as plt

sys.path.append('/data3/home/mbito/project_dynamic_graphs/tensorflow/evaluation')
import utils

class SIDynamic(): 
    def __init__(self, start_graphs, inf_alpha=1, inf_beta=1, sus_alpha=1, sus_beta=1): 
        '''
        inf_alpha: alpha parameter for infectious feature
        inf_beta: beta parameter for infectious feature
        '''
        # graph parameters
        self.timesteps = len(start_graphs)
        
        # infection parameters
        self.inf_alpha, self.inf_beta = inf_alpha, inf_beta
        self.sus_alpha, self.sus_beta = sus_alpha, sus_beta
        self.N_INIT_NODES = 1
        
        self.start_graphs = start_graphs
        self.graphs = self.simulate_infection(self.start_graphs)
        
    def initialize_graph(self, g):
        infected_nodes = np.random.choice(g.nodes, size=(self.N_INIT_NODES, ))
        for node in g.nodes: 
            infectious = np.random.beta(self.inf_alpha, self.inf_beta)
            susceptible = np.random.beta(self.sus_alpha, self.sus_beta)
            if node in infected_nodes:
                attrs = {'positive': 1, 'exposure': 1, 'susceptible': susceptible, 'infectious': infectious}
                for key in attrs: 
                    g.nodes[node][key] = attrs[key]
            else:
                attrs = {'positive': 0, 'exposure': 0, 'susceptible': susceptible, 'infectious': infectious}
                for key in attrs: 
                    g.nodes[node][key] = attrs[key]

        return g

    def simulate_infection(self, start_graphs): 
        while True: 
            graphs = []
            graph_history = nx.Graph()
            for i in range(self.timesteps): 
                if i == 0: 
                    g = self.initialize_graph(start_graphs[i].copy())
                else: 
                    g = start_graphs[i].copy()

                    # copy data from previous timestep
                    for node in g.nodes: 
                        if node in graph_history.nodes: 
                            for key in graph_history.nodes[node].keys(): 
                                g.nodes[node][key] = graph_history.nodes[node][key]
                        else:
                            infectious = np.random.beta(self.inf_alpha, self.inf_beta)
                            susceptible = np.random.beta(self.sus_alpha, self.sus_beta)
                            attrs = {'positive': 0, 'exposure': 0, 'timestep': 0, 'susceptible': susceptible, 'infectious': infectious}
                            for key in attrs.keys(): 
                                g.nodes[node][key] = attrs[key]

                g = self.generate_features(self.simulate_infection_single(g))
                graphs.append(g)
                graph_history.add_nodes_from(g.nodes(data=True))
                
            if utils.sis_infected_rate(graphs[-1]) > utils.sis_susceptible_rate(graphs[-1]): 
                break
            
        return graphs

    def simulate_infection_single(self, g):
        next_g = g.copy()
        
        # update next_g attributes: set nodes with exposure = 1 to positive = 1
        for node in next_g.nodes: 
            if next_g.nodes[node]['exposure'] == 1: 
                next_g.nodes[node]['positive'] = 1
        
        for node in next_g.nodes:
            if g.nodes[node]['exposure'] == 0:
                infected = 0
                p_uninfected = 1
                for neighbor in g.neighbors(node): 
                    p_neighbor = g.nodes[node]['susceptible'] * g.nodes[neighbor]['infectious'] * g.nodes[neighbor]['exposure']
                    p_uninfected *= 1 - p_neighbor
                    infected += np.random.binomial(1, p_neighbor)
                    
                if infected > 0: 
                    next_g.nodes[node]['exposure'] = 1
                    next_g.nodes[node]['p_infected'] = 1 - p_uninfected
                else: 
                    next_g.nodes[node]['exposure'] = 0
                    next_g.nodes[node]['p_infected'] = 1 - p_uninfected
            else: 
                next_g.nodes[node]['p_infected'] = -1 # for debugging purposes

        return next_g

    def generate_features(self, g): 
        for node in g.nodes: 
            features = []
            if g.nodes[node]['positive'] == 1: 
                features.append(1)
                features.append(g.nodes[node]['infectious'])
                features.append(g.nodes[node]['susceptible'])
            else: 
                features.append(-1)
                features.append(g.nodes[node]['infectious'])
                features.append(g.nodes[node]['susceptible'])

            g.nodes[node]['features'] = np.array(features)

        return g
    
class SISDynamic(): 
    def __init__(self, start_graphs, patience=10, inf_alpha=1, inf_beta=1, sus_alpha=1, sus_beta=1, rec_alpha=1, rec_beta=1): 
        '''
        inf_alpha: alpha parameter for infectious feature
        inf_beta: beta parameter for infectious feature
        '''
        # graph parameters
        self.timesteps = len(start_graphs)
        self.patience = patience
        
        # infection parameters
        self.inf_alpha, self.inf_beta = inf_alpha, inf_beta
        self.sus_alpha, self.sus_beta = sus_alpha, sus_beta
        self.rec_alpha, self.rec_beta = rec_alpha, rec_beta
        self.N_INIT_NODES = 1
        
        self.start_graphs = start_graphs
        self.graphs = self.simulate_infection(self.start_graphs)
        
    def initialize_graph(self, g):
        infected_nodes = np.random.choice(g.nodes, size=(self.N_INIT_NODES, ))
        for node in g.nodes: 
            infectious = np.random.beta(self.inf_alpha, self.inf_beta)
            susceptible = np.random.beta(self.sus_alpha, self.sus_beta)
            recovery = np.random.beta(self.rec_alpha, self.rec_beta)
            if node in infected_nodes:
                attrs = {'positive': 1, 'exposure': 1, 'susceptible': susceptible, 'recovery': recovery, 'infectious': infectious}
                for key in attrs: 
                    g.nodes[node][key] = attrs[key]
            else:
                attrs = {'positive': 0, 'exposure': 0, 'susceptible': susceptible, 'recovery': recovery, 'infectious': infectious}
                for key in attrs: 
                    g.nodes[node][key] = attrs[key]

        return g

    def simulate_infection(self, start_graphs): 
        for iterations in range(self.patience): 
            graphs = []
            graph_history = nx.Graph()
            for i in range(self.timesteps): 
                if i == 0: 
                    g = self.initialize_graph(start_graphs[i].copy())
                else: 
                    g = start_graphs[i].copy()
                    old = 0
                    new = 0

                    # copy data from previous timestep
                    for node in g.nodes: 
                        if node in graph_history.nodes:
                            for key in graph_history.nodes[node].keys(): 
                                g.nodes[node][key] = graph_history.nodes[node][key]
                        else:
                            infectious = np.random.beta(self.inf_alpha, self.inf_beta)
                            susceptible = np.random.beta(self.sus_alpha, self.sus_beta)
                            recovery = np.random.beta(self.rec_alpha, self.rec_beta)
                            attrs = {'positive': 0, 'exposure': 0, 'timestep': 0, 'susceptible': susceptible, 'recovery': recovery, 'infectious': infectious}
                            for key in attrs.keys(): 
                                g.nodes[node][key] = attrs[key]

                g = self.generate_features(self.simulate_infection_single(g))
                graphs.append(g)
                graph_history.add_nodes_from(g.nodes(data=True))
                
            if utils.sis_infected_rate(graphs[-1]) > 0:
                break
            
        return graphs

    def simulate_infection_single(self, g):
        next_g = g.copy()
        
        # update next_g attributes: set nodes with exposure = 1 to positive = 1
        for node in next_g.nodes: 
            if next_g.nodes[node]['exposure'] == 1: 
                next_g.nodes[node]['positive'] = 1
        
        for node in next_g.nodes:
            if g.nodes[node]['exposure'] == 0:
                infected = 0
                p_uninfected = 1
                for neighbor in g.neighbors(node): 
                    p_neighbor = g.nodes[node]['susceptible'] * g.nodes[neighbor]['infectious'] * g.nodes[neighbor]['exposure']
                    p_uninfected *= 1 - p_neighbor
                    infected += np.random.binomial(1, p_neighbor)
                    
                if infected: 
                    next_g.nodes[node]['exposure'] = 1
                    next_g.nodes[node]['p_infected'] = 1 - p_uninfected
                else: 
                    next_g.nodes[node]['exposure'] = 0
                    next_g.nodes[node]['p_infected'] = 1 - p_uninfected
            else: 
                recovered = np.random.binomial(1, g.nodes[node]['recovery'])
                if recovered: 
                    next_g.nodes[node]['exposure'] = 0
                    next_g.nodes[node]['positive'] = 0
                    next_g.nodes[node]['p_infected'] = -1
                else: 
                    next_g.nodes[node]['p_infected'] = -1

        return next_g

    def generate_features(self, g): 
        for node in g.nodes: 
            features = []
            if g.nodes[node]['positive'] == 1: 
                features.append(1)
                features.append(g.nodes[node]['infectious'])
                features.append(g.nodes[node]['susceptible'])
                features.append(g.nodes[node]['recovery'])
            else: 
                features.append(-1)
                features.append(g.nodes[node]['infectious'])
                features.append(g.nodes[node]['susceptible'])
                features.append(g.nodes[node]['recovery'])

            g.nodes[node]['features'] = np.array(features)

        return g