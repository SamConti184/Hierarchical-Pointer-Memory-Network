import tensorflow as tf
from tensorflow.keras import layers
from LuongAttention import LuongAttention
from WordAttention import WordAttention
from tensorflow.keras import initializers

#The original paper lacks many implementation details when describing the Decoder
#of the architecture. I will show and explain some of the assumptions that I made.
#Probability of generate word from the dictionary and from copy OOV words.
class Decoder(layers.Layer):
    def __init__(self, vocab_size, dec_units, batch_size, oov_max_length):
        super(Decoder, self).__init__()
        self.oov_max_length = oov_max_length
        self.batch_size = batch_size
        self.vocab_size = vocab_size
        self.dec_units = dec_units #Units of the GRU component
        #Each target word is embedded and then passed to the GRU
        self.gru = tf.keras.layers.GRU(self.dec_units,
                                        recurrent_initializer='glorot_uniform')
        #Dense Layer for producing a distribution over the word in the
        #dictionary. Activation is softmax: we will create a distribution over
        #the vocabulary: we will specify in the CrossEntropy not to consider logits,
        #but probabilities (so that we can combine this generation probability with
        #the copy distribution)
        self.fc = tf.keras.layers.Dense(vocab_size, activation="softmax")
        #Attention Component over the encoded sentences
        self.sentences_attention = LuongAttention(self.dec_units, batch_size)
        self.word_attention = WordAttention(self.dec_units)
        self.kernel_initializer = initializers.get('glorot_uniform')

    #The learnable weights used in the definition of the soft gate p_gen.
    #Vectors that will be used in dot products with the context vector,
    #the hidden state fo the decoder, and the input of the decoder.
    def build(self, input_shape):
        self.weight_context = self.add_weight(name="weight_context",
                    shape=(self.dec_units, 1),
                    initializer=self.kernel_initializer,
                    trainable=True) #shape (GRU*2, 1)
        self.weight_hidden = self.add_weight(name="weight_hidden_state",
                    shape=(self.dec_units, 1),
                    initializer=self.kernel_initializer,
                    trainable=True) #shape (GRU*2, 1)
        self.weight_dec = self.add_weight(name="weight_decoder_input",
                    shape=(self.dec_units*2, self.dec_units//2),
                    initializer=self.kernel_initializer,
                    trainable=True) #shape (GRU*3, 1)
        self.weight_input = self.add_weight(name="Attention_words",
                    shape=(self.dec_units//2, 1),
                    trainable=True)
        #super(Decoder, self).build(input_shape)

    def change_batch_size(self, batch_size):
        self.batch_size = batch_size


    #We assume to operate on a single batch -> so the shapes are:
    #embedded_target -> (batch_size x GRU) -> the current target word embedded
    #hidden_dec -> (batch_size, GRU*2) -> the hidden state, initially the query produced by the Encoder
    #enc_sentences -> (batch_size x num_sentences x GRU_units*2)
    #encoded_words -> (batch_size x num_sentences x num_words, GRU_units*2)
    #numeric_input -> (batch_size, num_sentences*len_sentences)
    def call(self, embedded_target, hidden_dec, encoded_sentences, encoded_words, numeric_input, batch_size):
        #initialize the GRU with the previous hidden state
        dec_output = self.gru(embedded_target, initial_state=hidden_dec)
        context, alpha = self.sentences_attention([dec_output, encoded_sentences])
        #Concatenation of the context vector and the output of the GRU
        # (batch_size x GRU*2); (batch_size x GRU*1)
        x = tf.concat([tf.squeeze(context, 1), dec_output], axis=-1) #batch_size x GRU*3
        hidden_t = tf.nn.tanh(tf.matmul(x, self.weight_dec)) #batch_size x GRU*4
        gen_distr = self.fc(hidden_t) #gen_distr has shape (batch_size, vocab_dim)
        #Padding the distribution up to (batch_size x vocab_dim + max_oov_length)
        #to sum this distribution with the copy distribution
        #copy distribution has size of vocabulary dimension + max number of oov words
        gen_distr = tf.pad(gen_distr, [[0,0], [0, self.oov_max_length]])

        #Obtaining beta, the distribution over each word
        #(batch_size, num_sentences, len_sentences)
        beta = self.word_attention([dec_output, encoded_words, alpha], batch_size)
        #Creation of the copy distribution: the distribution is over the vocabulary
        #and the words in memory. for each word in memory we check their vocabulary ID,
        #if they are OOV word their ID is vocab_dim + position in oov_dictionary.
        #Making beta as (batch_size, num_sentences * len_sentences)
        beta = tf.reshape(beta, (batch_size, -1))
        #Making numeric input as (batch_size, num_sentences*len_sentences, 1)
        numeric_input = tf.expand_dims(numeric_input, axis=-1)
        #Scatter_nd to sum beta values over the extended distribution
        op = lambda x: tf.scatter_nd(x[0], x[1], tf.constant([self.vocab_size + self.oov_max_length]))
        #For each numeric ID in the input we sum the beta values
        copy_distr = tf.map_fn(op, (numeric_input, beta), dtype=(tf.float32))

        p_gen = self.get_generation_gate_term(context, dec_output, embedded_target, batch_size)

        return dec_output, gen_distr, copy_distr, p_gen

    #While the definition of this gate term is not explicit in the paper,
    #it's described in another cited paper as shown in my main text work.
    #The returned term is a soft gate which defines the weight of the generative
    #probability distribution over the vocabulary. It's inverse is used to
    #weight the copying probability over the terms in the input.
    #context_vector -> (batch_size x GRU*2)
    #hidden_state of the decoder -> (batch_size, GRU*2)
    #input, the decoder input (embedded previous target word) -> (batch_size x GRU*2)
    def get_generation_gate_term(self, context, hidden_state, input, batch_size):
        #Dot product between the context vector and its weight
        #(batch_size, GRU*2) x (GRU*2, 1) -> (batch_size, 1)
        res1 = tf.matmul(tf.squeeze(context, 1), self.weight_context)
        #Dot product between the hidden state and its weight
        #(batch_size, GRU*2) x (GRU*2, 1) -> (batch_size, 1)
        res2 = tf.matmul(hidden_state, self.weight_hidden)
        #Reshaping the input to -> (batch_size, GRU*2)
        #Dot product between the input and its weight
        #(batch_size, GRU*2) x (GRU*2, 1) -> (batch_size, 1)
        res3 = tf.matmul(tf.reshape(input, [batch_size, self.dec_units//2]), self.weight_input)
        sum = res1 + res2 + res3
        p_gen = tf.nn.sigmoid(sum)    #shape -> (batch_size, 1)
        return p_gen    #Returned the p_gelue for each element in the batch.
