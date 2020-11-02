import tensorflow as tf
from tensorflow.keras import layers
from tensorflow.keras import initializers
#A Keras model used to evaluate intermediate results over the words embedding
#of each sentence of each element of the batch.
class WordAttention(layers.Layer):
    def __init__(self, weight_shape):
        super(WordAttention, self).__init__()
        self.weight_shape = weight_shape
        #self.batch_size = batch_size
        self.kernel_initializer = initializers.get('glorot_uniform')

    def build(self, input_shape):
        self.W_words = self.add_weight(name="Attention_words",
                    shape=(self.weight_shape, self.weight_shape),
                    initializer=self.kernel_initializer,
                    trainable=True)
        #super(WordAttention, self).build(input_shape)

    #words is a 4d input of shape (batch_size, num_sentences, len_sentence, GRU*2)
    #hidden state is a 2D input of shape (batch_size, GRU*2)
    #alpha -> #(batch_size x 1 x num_sentences)
    #Word attention is defined in the original Hierarchical Pointer MMN
    def call(self, inputs, batch_size):
        hidden = inputs[0]
        words = inputs[1]
        alpha = tf.cast(inputs[2], tf.float32)
        #We need to evaluate the attention value for each word.
        #Firstly we reduce the tensor words to 3 dimensions.
        #Now words is of shape (batch_size, num_sentences*len_sentences, GRU*2)
        #We then evaluate an intermediate result given by the dot product
        #between the weight matrix and the hidden state
        #(batch_size, GRU*2) x (GRU*2, GRU*2) -> (batch_size, GRU*2)
        inter_res = tf.matmul(hidden, self.W_words)
        #We then need to perform a dot product between each encoded word
        #and the intermediate result.
        #We define the operation as
        op = lambda a: tf.matmul(tf.reshape(a[0], [1, -1]), a[1], transpose_b=True)
        #Shape of result is (batch_size x num_sentences x 1 x len_sentences)
        res = tf.map_fn(op, (inter_res, words), dtype=(tf.float32))
        #We then evaluate the softmax over the axis of each sentence
        #words = tf.math.maximum(words, 1e-5)
        #(batch_size x num_sentences x 1 x len_sentences)
        e = tf.nn.softmax(res, axis=-1)
        #We then evaluate the value beta
        #beta has shape (batch_size, num_sentences, len_sentences, 1)
        #For each value now we have the corrispondent beta value
        alpha = tf.squeeze(alpha, 1)
        alpha = tf.expand_dims(alpha, axis=-1) #(batch_size, num_sentences, 1)
        beta = alpha * tf.squeeze(e) ##(batch_size, num_sentences, 1) * (batch_size, num_sentences, len_sentences)
        #(batch_size, num_sentences, len_sentences, 1)
        return beta
