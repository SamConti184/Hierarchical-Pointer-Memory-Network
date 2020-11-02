import tensorflow as tf
from tensorflow.keras import layers


#Definition of a MemN2N encoder as based on the version proposed in the paper,
#which presents some differences with the version shown in the Sukhabaatar paper.
class MemN2N(layers.Layer):
    def __init__(self, hops, weight, batch):
        super(MemN2N, self).__init__()
        self.hops = hops
        self.weight = weight
        self.batch_size = batch

    #Layer-wise implementation. In this version only one weight of matrices is used,
    #instead of two per hop as in the original paper.
    def build(self, input_shape):
        self.W = self.add_weight(name="MemN2N_kernel",
                    shape=(self.weight, self.weight),
                    trainable=True)
        super(MemN2N, self).build(input_shape)

    #Input of (batch_size, num_sentences, GRU_units*2))
    #Masking (batch_size, num_sentences)
    def call(self, sentences, masking):
        def operation():
            def op(inp):
                input = inp[0] #(num_sentences, GRU_units*2))
                masking = inp[1] #(1, num_sentences)
                query = inp[2] #(1, GRU_units*2)
                #Evaluate the last valid sentences of the input (last non completely padded)
                new_mask = tf.reduce_sum(tf.cast(masking, tf.int32))
                op1 = lambda a: tf.matmul(a[0], a[1], transpose_b=True) #query*sentence
                if(new_mask != 0):
                    cur_input = input[:new_mask-1, :]
                else:
                    cur_input = input[:1, :]
                for k in range(self.hops):
                    res = tf.matmul(cur_input, query, transpose_b=True) #A column vector of shape (num_sentences-1, 1)
                    p = tf.nn.softmax(res, axis=1) #(num_sentences-1, 1)
                    #p * input -> (num_sentences-1, GRU_Units)
                    x = p*cur_input
                    #(1 x GRU_Units*2) x (GRU_units*2 x GRU_Units*2)
                    x = tf.math.reduce_sum(x, axis=0, keepdims=True)
                    o= tf.matmul(x, self.W) #(1 , GRU_units*2)
                    query = tf.math.add(o, query)
                return query
            return op
        #Selecting the query as the last valid sentence before padding
        query = tf.map_fn(lambda x: x[0][tf.reduce_sum(tf.cast(x[1], tf.int32)) - 1, :], (sentences, masking), dtype=tf.float32)
        query = tf.expand_dims(query, axis=1) #(batch_size, 1, 256)
        query = tf.map_fn(operation(), (sentences, masking, query), dtype=tf.float32)
        return query
