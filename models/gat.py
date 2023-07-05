import tensorflow as tf
import numpy as np
from sklearn.metrics import roc_auc_score

class GraphAttention(tf.keras.layers.Layer): 
    def __init__(self, units, heads, activation=tf.nn.relu, dropout=.2, bias=0, final_layer=False):
        super(GraphAttention, self).__init__()
        self.units = units
        self.heads = heads
        self.activation = activation
        self.dropout = dropout
        self.bias = bias
        self.final_layer = final_layer
        
    def build(self, input_shape): 
        self.w = self.add_weight(shape=(input_shape[-1], self.units), initializer='glorot_normal', trainable=True)
        if self.bias: 
            self.b = self.add_weight(shape=(self.units, ), initializer='glorot_normal', trainable=True)
        
        self.attn_w1 = self.add_weight(shape=(self.units, self.heads), initializer='glorot_normal', trainable=True)
        self.attn_w2 = self.add_weight(shape=(self.units, self.heads), initializer='glorot_normal', trainable=True)
        
    def call(self, inputs, adj): 
        x = inputs
        
        if self.dropout: 
            x = tf.nn.dropout(x, self.dropout)
        
        x = tf.linalg.matmul(x, self.w)
        if self.bias: 
            x = tf.math.add(x, self.b)
            
        a1 = tf.expand_dims(tf.linalg.matmul(x, self.attn_w1), 1) # a1.shape --> (N, 1, n_heads)
        a2 = tf.expand_dims(tf.linalg.matmul(x, self.attn_w2), 1) # a2.shape --> (N, 1, n_heads)
        # coeffs = tf.nn.sigmoid(tf.math.add(a1, tf.transpose(a2, perm=[1, 0, 2]))) # softmax(relu(a1 + a2^T))
        coeffs = tf.nn.relu(tf.math.add(a1, tf.transpose(a2, perm=[1, 0, 2]))) # softmax(relu(a1 + a2^T))
        
        attns = []
        for i in range(self.heads):
            # **for sparse matrix multiply**
            # attn_adj = tf.sparse.from_dense(tf.math.multiply(adj, coeffs[:, :, i]))
            # attns.append(tf.sparse.to_dense(tf.sparse.sparse_dense_matmul(attn_adj, x)))
            # attn_adj = tf.sparse.from_dense(attn_adj) 
            attn_adj = tf.nn.softmax(tf.math.multiply(adj, coeffs[:, :, i]), axis=1)
            # attn_adj = tf.math.multiply(adj, coeffs[:, :, i])
            attns.append(tf.linalg.matmul(attn_adj, x))
            
        # todo: add residual connections
        attns = tf.concat(attns, axis=-1)
        
        if self.final_layer: 
            attns = tf.math.reduce_mean(attns, axis=-1)
        
        attns = self.activation(attns)
        
        return attns
    
    def get_attn_coefs(self, inputs, adj): 
        x = inputs
        
        x = tf.linalg.matmul(x, self.w)
        
        if self.bias: 
            x = tf.math.add(x, self.b)
            
        a1 = tf.expand_dims(tf.linalg.matmul(x, self.attn_w1), 1) # a1.shape --> (N, 1, n_heads)
        a2 = tf.expand_dims(tf.linalg.matmul(x, self.attn_w2), 1) # a2.shape --> (N, 1, n_heads)
        # coeffs = tf.nn.sigmoid(tf.math.add(a1, tf.transpose(a2, perm=[1, 0, 2]))) # softmax(relu(a1 + a2^T))
        coeffs = tf.nn.relu(tf.math.add(a1, tf.transpose(a2, perm=[1, 0, 2]))) # softmax(relu(a1 + a2^T))
            
        attns = []
        for i in range(self.heads):
            # **for sparse matrix multiply**
            # attn_adj = tf.sparse.from_dense(tf.math.multiply(adj, coeffs[:, :, i]))
            # attns.append(tf.sparse.to_dense(tf.sparse.sparse_dense_matmul(attn_adj, x)))
            # attn_adj = tf.sparse.from_dense(attn_adj) 
            attn_adj = tf.expand_dims(tf.nn.softmax(tf.math.multiply(adj, coeffs[:, :, i]), axis=1), axis=2)
            # attn_adj = tf.expand_dims(tf.math.multiply(adj, coeffs[:, :, i]), axis=2)
            attns.append(attn_adj)
            
        # todo: add residual connections
        attns = tf.concat(attns, axis=-1)
        
        if self.final_layer: 
            attns = tf.math.reduce_mean(attns, axis=-1)
            
        return attns
    
class GAT(tf.keras.Model): 
    def __init__(self, num_layers, n_classes, bias, dropout, optimizer, loss): 
        super(GAT, self).__init__()
        self.num_layers = num_layers
        self.n_classes = n_classes
        self.bias = bias
        self.dropout = dropout
        self.optimizer = optimizer
        self.loss = loss
        self.model_layers = []
        
        for layers in range(self.num_layers - 1):
            self.model_layers.append(GraphAttention(units=16, heads=4, dropout=self.dropout, bias=self.bias))
        self.model_layers.append(GraphAttention(units=1, heads=1, dropout=self.dropout, bias=self.bias, activation=tf.nn.sigmoid, final_layer=True))
        
    def call(self, inputs, adj): 
        x = inputs
        for i in range(len(self.layers)):
            x = self.layers[i](inputs=x, adj=adj)
            
        return x
    
    @tf.function
    def train_step(self, data):
        X, y, y_mask, adj, adj_mean, adj_norm = data
        y_mask = tf.cast(y_mask, tf.bool)
        with tf.GradientTape() as tape:
            y_hat = self(inputs=X, adj=adj_norm, training=True)
            y_hat = tf.boolean_mask(y_hat, y_mask)
            y = tf.boolean_mask(y, y_mask)
            loss_value = self.loss(y, y_hat)
            
        grads = tape.gradient(loss_value, self.trainable_weights)
        self.optimizer.apply_gradients(zip(grads, self.trainable_weights))
        
        return y, y_hat
    
    @tf.function
    def test_step(self, data):
        X, y, y_mask, adj, adj_mean, adj_norm = data
        y_mask = tf.cast(y_mask, tf.bool)
        y_hat = self(inputs=X, adj=adj_norm, training=False)
        y_hat = tf.boolean_mask(y_hat, y_mask)
        y = tf.boolean_mask(y, y_mask)
        
        return y, y_hat
        
        
    def train(self, train_dataset, valid_dataset, epochs=10, early_stopping=3, verbose=False):
        early_stopping_count = 0
        best_loss = 100
        train_bce = tf.keras.metrics.BinaryCrossentropy()
        valid_bce = tf.keras.metrics.BinaryCrossentropy()
        
        for epoch in range(epochs): 
            for step, data in enumerate(train_dataset):
                y, y_hat = self.train_step(data)
                train_bce.update_state(y, y_hat)
            
            # early stopping
            for step, data in enumerate(valid_dataset):
                y, y_hat = self.test_step(data)
                valid_bce.update_state(y, y_hat)
            
            loss_value = float(valid_bce.result())
            if loss_value < best_loss - .005: 
                best_loss = loss_value
                early_stopping_count = 0
            else: 
                early_stopping_count+=1
                    
            if early_stopping_count == early_stopping: 
                if verbose:
                    print(f'Validation loss has not increased for {early_stopping} epochs')
                break
                
            if verbose:
                print(f'Epoch {epoch}: Training loss is {float(train_bce.result()):.3f}, Validation loss is {float(valid_bce.result()):.3f}')
            
            train_bce.reset_states()
            valid_bce.reset_states()
            
    def evaluate(self, valid_dataset, test_dataset, verbose=False): 
        val_auc = tf.keras.metrics.AUC()
        test_auc = tf.keras.metrics.AUC()
        aucs = []
        preds = []
        
        for step, data in enumerate(valid_dataset):
            y, y_hat = self.test_step(data)
            val_auc.update_state(y, y_hat)
        
        for step, data in enumerate(test_dataset):
            y, y_hat = self.test_step(data)
            if tf.unique(y)[0].shape[0] > 1:
                test_auc.update_state(y, y_hat)
                aucs.append(roc_auc_score(y, y_hat))
            else: 
                aucs.append(np.inf)
            preds.append(y_hat)
            
        aucs = np.array(aucs)
        val_auc = float(val_auc.result())
        test_auc = float(test_auc.result())
        test_error = 1.96 * np.std(aucs[np.isinf(aucs) == 0]) / np.sqrt(aucs[np.isinf(aucs) == 0].shape[0])
        
        if verbose:
            print(f'GCN AUC on the test set is {auc:.4f}')
        
        return test_auc, test_error, aucs, val_auc, preds
    
    def get_attn_coefs(self, test_dataset): 
        def process_attns(attn): 
            proc_attn = np.zeros((attn.shape[1], ))
            
            for i in range(attn.shape[0]): 
                mask = (attn[:, i] > 0).astype(bool)
                proc_attn[i] = np.mean(attn[:, i][mask])
            
            return proc_attn
        
        total_attns = []
        for step, data in enumerate(test_dataset):
            x, y, y_mask, adj = data
            
            for i in range(len(self.layers) - 1):
                x = self.layers[i](inputs=x, adj=adj)
                
            if len(self.layers) == 1:
                attns = self.layers[0].get_attn_coefs(inputs=x, adj=adj).numpy()
            else: 
                attns = self.layers[i + 1].get_attn_coefs(inputs=x, adj=adj).numpy()
                
            attns = attns[y_mask.numpy().astype(bool)] # automatically broadcasts
            attns = process_attns(attns) 
            # attns = np.max(attns, axis=0)
            
            total_attns.append((attns, y_mask))
        
        return total_attns