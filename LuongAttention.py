import tensorflow as tf
from tensorflow.keras import layers
from tensorflow.keras import initializers
#Luong Attention - global attention with general score
#implemented as from the original Luong paper
class LuongAttention(layers.Layer):
        def __init__(self, rnn_size, batch_size):
            super(LuongAttention, self).__init__()
            self.batch_size = batch_size
            self.rnn_size = rnn_size    #GRU*2
            self.kernel_initializer = initializers.get('glorot_uniform')

        def build(self, input_shape):
            self.Wa = self.add_weight(name="Attention_sent",
                        shape=(self.rnn_size, self.rnn_size),
                        initializer=self.kernel_initializer,
                        trainable=True)
            #super(WordAttention, self).build(input_shape)

         #Hidden state is the state of the GRu in the Decoder
         #sentences are the sentence saved in memory batch x num_sentences x GRU_units*2
        def call(self, inputs):
            dec_output = inputs[0] #(batch x GRU)
            sentences = inputs[1]  #(batch, num_sentences, GRU*2)
            #State of the target produced by decoder now is (batch_size, 1, GRU*2)
            dec_output = tf.expand_dims(dec_output, axis=1)
            #tf.print(dec_output.shape)
            #score has dimension (batch, 1, num_sentences)
            score = tf.matmul(dec_output, tf.matmul(sentences, self.Wa), transpose_b=True)
            #tf.print(score.shape)
            #(batch, 1, num_sentences)
            alpha = tf.nn.softmax(score, axis=2)
            #tf.print(alpha.shape)
            #(batch, 1, GRU*2)
            context = tf.matmul(alpha, sentences)
            #tf.print(sentences.shape)
            #tf.print(context.shape)
            return context, alpha
