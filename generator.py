import pickle as pkl
import tensorflow as tf
import numpy as np
import os

class generator():
    def __init__(self, split, data_path):
        self.split = split
        self.data_path = data_path
        self.max_idx = len(os.listdir(os.path.join(self.data_path, self.split)))

    def data_generator(self, ):
        for i in range(self.max_idx):
            with open(os.path.join(self.data_path, self.split, f'g_{i}.pkl'), 'rb') as file:
                X, y, y_mask, adj, adj_mean, adj_norm = pkl.load(file)

            yield X, y, y_mask, adj, adj_mean, adj_norm

    def tf_generator(self, generator): 
        tf_gen = tf.data.Dataset.from_generator(generator, output_types=(tf.float32, tf.float32, tf.float32, tf.float32, tf.float32, tf.float32))
        tf_gen = tf_gen.prefetch(tf.data.experimental.AUTOTUNE)
        tf_gen = tf_gen.take(self.max_idx)

        return tf_gen