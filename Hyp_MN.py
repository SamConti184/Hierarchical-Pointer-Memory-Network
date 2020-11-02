#The model coded in this module is the Hierarchical-Pointer Generator Memory Network.
#Tensorflow 2.2 is the main framework used for creating, training and testing the model.
#The data used to create the Dataset from TensorFlow is obtained by the module
#Data_preporcessing, but can be performed as preferreds.

import tensorflow as tf
import numpy as np
import os.path
import time
from tensorflow import keras
from tensorflow.keras import layers
from Data_preprocessing import Dataset
from WordAttention import WordAttention
from LuongAttention import LuongAttention
from Decoder import Decoder
from MemN2N import MemN2N

#Simple dictionary getting the name corresponding to the task for the bAbI dataset
def babi_task_name(task):
    return {1 : "API-calls",
            2 : "API-refine",
            3 : "options",
            4 : "phone-address",
            5 : "full-dialogs",
            6 : "dstc2"}.get(task, "dstc2")

#Function used to retrieve from an external Numpy dataset
#the already processed bAbI dataset, properly divided in features, target, etc.
def get_features_labels(task, mode, oov):
    filename = "/dialog-bAbI-tasks/dialog-bAbI-task" + \
                            str(task) + "-" + babi_task_name(task) + "-"
    if(mode == "training"):
        filename = filename + "trn.txt.npz"
    elif(mode == "develop"):
        filename = filename + "dev.txt.npz"
    #If oov is True and we search for a testing set, we retrieve the testing OOV set.
    elif(oov):
        filename = filename + "tst-OOV.txt.npz"
    else:
        filename = filename + "tst.txt.npz"
    with np.load("./Datasets/Preprocessed" + filename, allow_pickle=True) as data:
        features = data["x1"] #the tokenized input dialogues, numerical IDs inputs
        sentences_lengths = data["x2"] #actual lenght of each sentence in each dialogue
        input_IDs = data["x3"] #modified features IDs, introducing different numerical indices for OOV words.
        input_text = data["x4"] #input data in textual form
        labels_text = data["y1"] #target sentence in textual form
        system_IDs = data["y2"] #modified target IDs, introducing different numerical indices for OOV words.
        labels_IDs = data["y3"] #target IDs, numerical form, OOV IDs still = 1
    return features, input_IDs, labels_text, system_IDs, labels_IDs, sentences_lengths


#MODIFY the following values to try different sets of hyper-parameters
task = 5 #The task for the bAbI dataset
#Creation of a Dataset object
dataset = Dataset("babi", True, 0.1, task)
#Creation of a training and validation set in terms of Numpy arrays serialized
dataset.create_preprocessed_dataset(mode="training", oov=False)
dataset.create_preprocessed_dataset(mode="develop", oov=False)
vocab_dim = 0
oov_max_length = 0
embedding_dim = 256 #Dimension of the embedding layer
dec_units = 256 #Dimension of hidden units in GRUs
hops = 3 #number of hops for MemNN architecture in the Encoder
batch_size = 16 #elements per batch
epochs = 2 #number of epochs for training
rho = 1
gamma = 1
restore_weights = True #Set to True if we want to restore previously saved weights
lr = 0.001 #Learning rate of the Optimizer
batch_test_size = 1 #Size of the batch for the test set
train = False #Modify here to set if we are in training mode (weights need optimization)
              #or set to False for testing


#Creation of the TensorFlow Dataset from the .npz files
def create_dataset(dataset: Dataset, task: int, mode: str, oov: bool):
    features, input_IDs, labels_text, system_IDs, labels_id, sentences_lenghts = get_features_labels(task, mode, oov)
    vocab_dim, oov_max_length = dataset.get_important_dimensions()
    total_inputs = len(features)
    print(f'{vocab_dim} and {oov_max_length}\n')
    steps_per_epoch = total_inputs // batch_size -1
    max_sentence_len = len(features[0][0]) #getting the maximum length of the sentences in the dataset.
    max_len_target = len(labels_id[0])
    #Generator used for creating the Tensorflow dataset from serialized Numpy arrays.
    def generator():
        for el1, el2, el3, el4, el5, el6 in zip(features, input_IDs, labels_text, system_IDs, labels_id, sentences_lenghts):
            yield (el1, el2, el3, el4, el5, el6)
    #Constructing the Dataset from a generator based over the Numpy vectors.
    #the Dataset is then padded and repeated for the desired number of epochs.
    dataset = tf.data.Dataset.from_generator(generator,
            output_types=(tf.int32, tf.int32, tf.string, tf.int32, tf.int32, tf.int32),
            output_shapes=(tf.TensorShape([None, max_sentence_len]), tf.TensorShape([None]), tf.TensorShape([None]), tf.TensorShape([None]), tf.TensorShape([None]), tf.TensorShape([None])))
    #Padding and repeating for the number of epochs the dataset for either training or testing set
    if(mode == "training" or mode == "develop"):
        dataset = dataset.padded_batch(batch_size, padded_shapes=([None, max_sentence_len], [None], [None], [None], [None], [None]), padding_values= (0,0,"PAD",0,0,0), drop_remainder=True)
        dataset = dataset.repeat(epochs)
    else:
        dataset = dataset.padded_batch(batch_test_size, padded_shapes=([None, max_sentence_len], [None], [None], [None], [None], [None]), padding_values= (0,0,"PAD",0,0,0), drop_remainder=True)
    return dataset, steps_per_epoch, vocab_dim, oov_max_length

#Creation of the TensorFlow datasets
train_dataset, steps_per_epoch, vocab_dim, oov_max_length = create_dataset(dataset, task, "training", False)
validation_dataset, _, _, oov_max_length_valid = create_dataset(dataset, task, "develop", False)

#MODEL DEFINITION
#EMBEDDING LAYER DEFINITION
#Notice mask_zero=True to create a making matrix for the padding
embedding_layer = layers.Embedding(vocab_dim, embedding_dim, embeddings_initializer="glorot_uniform", mask_zero=True)
#ENCODER LAYER DEFINITION
encoder = MemN2N(2, dec_units*2, batch_size)
#DECODER LAYER DEFINITION
decoder = Decoder(vocab_dim, dec_units*2, batch_size, oov_max_length)

#Definition of a Custom loss in tensorflow.keras
#The underlying loss is based on a Cross-entropy loss, only with the addition of a
#custom soft gate to combine a generation and a copy distribution.
#gen_distr: (batch_size x vocab_extend)
#copy_distr: (batch_size x vocab_extend)
#p_gen: (batch_size x 1)
def custom_loss(gen_distr, copy_distr, p_gen):
    #p_gen is (batch_size, ), just a vector
    #p_gen is a soft-gate between 0 and 1 created by the model
    p_gen = tf.cast(p_gen, tf.float32)
    def loss(y_true):
        #mask for not considering the loss over the padding
        mask = tf.math.logical_not(tf.math.equal(y_true, 0))
        #Summation of the generative distribution and the copy distribution
        #through the soft-gate p_gen
        distr = p_gen*gen_distr + rho*(1-p_gen)*copy_distr
        loss_fun = tf.keras.losses.SparseCategoricalCrossentropy(from_logits=False)
        loss = loss_fun(tf.expand_dims(y_true, axis=-1), distr)
        loss = loss*tf.cast(mask, tf.float32)
        loss = tf.reduce_mean(loss)
        return loss
    return loss

#Definition of an accuracy function
#dist (batch_size, distr_size)
#y_true (batch_size, 1)
def train_accuracy(distr, y_true):
    y_true = tf.squeeze(tf.cast(y_true, tf.int32)) #(batch_size, )
    indices = tf.cast(tf.math.argmax(distr, axis=1), tf.int32) #(batch_size, )
    res1 = tf.math.equal(indices, y_true)
    res1 = tf.cast(res1, tf.float32)
    accuracy = tf.math.reduce_sum(res1)
    accuracy = accuracy / batch_size
    #tf.print(accuracy, summarize=-1)
    return accuracy

#Getting the most probable word for each batch
def eval(gen_distr, copy_distr, p_gen):
    final_distr = p_gen*gen_distr + rho*(1 - p_gen)*copy_distr
    #tf.print(final_distr)
    #tf.print(final_distr[0][tf.squeeze(true_value)])
    #tf.print(true_value)
    dec_output = tf.math.argmax(final_distr, axis=1, output_type=tf.int32)
    #tf.print(final_distr)
    return dec_output

#Definition of the Optimizer
optimizer = tf.keras.optimizers.Adam(learning_rate=lr)

#Definition of the GRU used in the encoding phase
gru = tf.keras.layers.GRU(units=dec_units, return_sequences=True)
#bi-gru concatenates by default the results of the forward and backward GRU
bi_gru = tf.keras.layers.Bidirectional(layer=gru)
#Concatenation operation for creating the encoding of each sentence:
#x[0] is the encoding produced by the bi-GRU, x[1] is the last word before padding
#[x[1], :dec_units] -> the last word of the sentence encoded by forward GRU
#[0, dec_units:] -> first word of the sentence encoded by backward GRU
op = lambda x: tf.concat([x[0][x[1], :dec_units], x[0][0, dec_units:]], axis=-1)
#x is a tuple containing the encoded words and the sentence lengths
#map_fn is a TensorFlow op which iterates over the batch dimension
op2 = lambda x: tf.map_fn(op, x, dtype=tf.float32)

#Definition of the directory in which saving the weights
checkpoint_dir = './training_checkpoints'
checkpoint_prefix = os.path.join(checkpoint_dir, str(task))
checkpoint = tf.train.Checkpoint(embedding_layer=embedding_layer, encoder=encoder, decoder=decoder, bi_gru=bi_gru)
print("End definition")


#This decorator is necessary to avoid the phenomeno of re-tracing associated
#with variable arguments for a function contained by tf.function
input_signature = [
        tf.TensorSpec(shape=(batch_size, None, None), dtype=tf.int32),
        tf.TensorSpec(shape=(batch_size, None), dtype=tf.int32),
        tf.TensorSpec(shape=(batch_size, None), dtype=tf.int32),
        tf.TensorSpec(shape=(batch_size, None), dtype=tf.int32),
        tf.TensorSpec(shape=(batch_size, None), dtype=tf.int32)
]
@tf.function(input_signature=input_signature)
def train_step(features, target, target_IDs, input_IDs, sentences_lengths):
    print("Built train function")
    loss = 0.0
    accuracy = 0.0
    with tf.GradientTape() as tape:
        #Definition of the Keras Model
        #Currently the Input x1 is a 3D Tensor. We need to apply Embedding to each
        #sentence of each batch.
        embed = embedding_layer(features)
        #Now the Input is in the shape of (32 x ? x max_len x embedding_dim)
        #We need to apply a Bidirectional GRU, whihc expèects a 3D input.
        #We again use the TimeDistributed wrapper.
        #We define a 4D tensors (32, num_sentences, len_sentences, GRU_units*2)
        #where each word is embedded as the concatenation of the corresponding hidden state
        #of the forward and backward GRU.
        encoded_words= layers.TimeDistributed(bi_gru)(embed)
        #tf.print(encoded_words.shape)
        #X_sentences is a 3D tensor of shape (32 x num_sentences x GRU_units*2)
        #Each sentence is represented by the last hidden state of the forward GRU
        #and the backward GRU
        #Computing padding mask from the bi-gru layer
        mask = layers.TimeDistributed(bi_gru).compute_mask(embed, embedding_layer.compute_mask(features))
        #Compute a new mask of dimension (batch_size x num_sentences) to indicate which sentences
        #are not composed by just padding symbols
        new_mask = tf.map_fn(lambda x: tf.reduce_any(x, axis=1), mask, dtype=tf.bool)
        encoded_sentences = layers.Lambda(lambda x: tf.map_fn(op2, (x[0], x[1]), dtype=tf.float32), mask=new_mask)([encoded_words, sentences_lengths])
        #tf.print(encoded_sentences.shape)
        #query_k (batch_size x 1 x GRU*2)
        query_k = encoder(encoded_sentences, masking=new_mask)

        #DECODING phase - Applying teacher forcing in training
        #query_k must be of dimension (batch_size x GRU*2) to be used as initial hidden state
        #Using target_IDs to maintain 1 for OOV words as input
        dec_hidden = tf.squeeze(query_k, axis=1)
        dec_input = embedding_layer(tf.expand_dims(target_IDs[:, 0], axis=-1))
        print("Starting loop")
        batch_accuracy = 0.0
        len_target = tf.shape(target)[1]
        #tf.print(len_target)
        #loop over each word in the target
        for t in range(1, len_target):
            #calling the Decoder
            dec_hidden, gen_distr, copy_distr, p_gen= decoder(dec_input, dec_hidden, encoded_sentences, encoded_words, input_IDs, batch_size)
            loss_f = custom_loss(gen_distr, copy_distr, p_gen)
            loss += loss_f(target[:, t])

            distr = p_gen*gen_distr + rho*(1-p_gen)*copy_distr
            batch_accuracy += train_accuracy(distr, target[:, t])
            #tf.print(batch_accuracy)

            #Embedding of the target word using the same embedding layer of the input
            #Teacher forcing
            dec_input = embedding_layer(tf.expand_dims(target_IDs[:, t], axis=-1))

    accuracy = batch_accuracy / (tf.cast(len_target, tf.float32) - 1.0)
    variables = encoder.trainable_variables + decoder.trainable_variables + embedding_layer.trainable_variables + bi_gru.trainable_variables
    gradients = tape.gradient(loss, variables)
    #gradients, _ = tf.clip_by_global_norm(gradients, 5.0)
    #tf.print(optimizer.learning_rate)
    optimizer.apply_gradients(zip(gradients, variables))
    return loss, accuracy


#Definition of a testing loop, pretty similar to the training one
test_input_signature = [
        tf.TensorSpec(shape=(batch_test_size, None, None), dtype=tf.int32),
        tf.TensorSpec(shape=(batch_test_size, None), dtype=tf.int32),
        tf.TensorSpec(shape=(batch_test_size, None), dtype=tf.int32),
        tf.TensorSpec(shape=(batch_test_size, None), dtype=tf.int32)
]
@tf.function(input_signature=test_input_signature)
def test_step(features, target, input_IDs, sentences_lengths):
    #target comprises OOV words with extra-vocabulary index: OOV words are not 1s.
    embed = embedding_layer(features)
    encoded_words= layers.TimeDistributed(bi_gru)(embed)
    mask = layers.TimeDistributed(bi_gru).compute_mask(embed, embedding_layer.compute_mask(features))
    new_mask = tf.map_fn(lambda x: tf.reduce_any(x, axis=1), mask, dtype=tf.bool)
    encoded_sentences = layers.Lambda(lambda x: tf.map_fn(op2, (x[0], x[1]), dtype=tf.float32), mask=new_mask)([encoded_words, sentences_lengths])
    query_k = encoder(encoded_sentences, masking=new_mask)
    dec_hidden = tf.squeeze(query_k, axis=1)
    dec_input = embedding_layer(tf.expand_dims(target[:, 0], axis=-1))

    dec_output = tf.constant([0])
    t = 0
    answer = tf.TensorArray(tf.int32, size=0, dynamic_size=True, clear_after_read=False)
    batch_accuracy = 0.0
    correct = 0
    while (dec_output != 3 and target[:, t] != 3):
        t = t+1
        #calling the previously trained Decoder
        #tf.print(dec_hidden)
        dec_hidden, gen_distr, copy_distr, p_gen= decoder(dec_input, dec_hidden, encoded_sentences, encoded_words, input_IDs, 1)
        #tf.print(p_gen)
        dec_output = eval(gen_distr, copy_distr, p_gen)
        if(dec_output == target[:,t]): #shapes: [1], [1]
            correct = 1
        else:
            correct = 0
        answer = answer.write(t-1, dec_output)
        #tf.print(dec_output)
        #If the word is an OOV, we pass UNK value  (1) for embedding
        if (dec_output < vocab_dim):
            dec_input = embedding_layer(tf.expand_dims(dec_output, axis=-1)) #NO teacher forcing
        else:
            dec_input = embedding_layer(tf.expand_dims(tf.constant([1]), axis=-1))
        if (correct == 1):
            batch_accuracy += 1.0
        dec_input = tf.reshape(dec_input, [1,1,dec_units])
        dec_output = tf.reshape(dec_output, [1])
    tf.print(batch_accuracy)
    tf.print(t)
    batch_accuracy = batch_accuracy / float(t)

    return batch_accuracy, answer.stack()

#Definition of a validation loop, similar to the training one, but no application
#of the gradient descent
@tf.function(input_signature=input_signature)
def validation_step(features, target, target_IDs, input_IDs, sentences_lengths):
    print("Called validation function")
    loss = 0.0
    accuracy = 0.0
    #Same as for training, but no modifications to weights and gradients
    embed = embedding_layer(features)
    encoded_words= layers.TimeDistributed(bi_gru)(embed)
    mask = layers.TimeDistributed(bi_gru).compute_mask(embed, embedding_layer.compute_mask(features))
    new_mask = tf.map_fn(lambda x: tf.reduce_any(x, axis=1), mask, dtype=tf.bool)
    encoded_sentences = layers.Lambda(lambda x: tf.map_fn(op2, (x[0], x[1]), dtype=tf.float32), mask=new_mask)([encoded_words, sentences_lengths])
    query_k = encoder(encoded_sentences, masking=new_mask)
    dec_hidden = tf.squeeze(query_k, axis=1)
    dec_input = embedding_layer(tf.expand_dims(target_IDs[:, 0], axis=-1))
    print("Starting loop")
    batch_accuracy = 0.0
    len_target = tf.shape(target)[1]
    #tf.print(len_target)
    for t in range(1, len_target):
        #calling the Decoder
        dec_hidden, gen_distr, copy_distr, p_gen= decoder(dec_input, dec_hidden, encoded_sentences, encoded_words, input_IDs, batch_size)
        loss_f = custom_loss(gen_distr, copy_distr, p_gen)
        loss += loss_f(target[:, t])

        distr = p_gen*gen_distr + rho*(1-p_gen)*copy_distr
        batch_accuracy += train_accuracy(distr, target[:, t])

        dec_input = embedding_layer(tf.expand_dims(target_IDs[:, t], axis=-1))

    accuracy = batch_accuracy / (tf.cast(len_target, tf.float32) - 1.0)
    return loss, accuracy


#TRAINING CASE:
if(train):
    if(restore_weights):
        checkpoint.restore(tf.train.latest_checkpoint(checkpoint_dir))
        print("Weights restored")
        optimizer = tf.keras.optimizers.Adam(learning_rate=0.0001) #Used for changing the learning rate
    train_iterator = iter(train_dataset)
    val_iterator = iter(validation_dataset)
    print("Starting training")
    for epoch in range(epochs):
        start = time.time()
        total_loss = 0
        total_validation_loss = 0
        for batch in range(steps_per_epoch):
            inp = next(train_iterator) #training
            inp2 = next(val_iterator)
            decoder.oov_max_length = oov_max_length
            batch_loss, accuracy = train_step(inp[0], inp[3], inp[4], inp[1], inp[5])
            total_loss += batch_loss
            decoder.oov_max_length = oov_max_length_valid
            validation_loss, validation_accuracy = validation_step(inp2[0], inp2[3], inp2[4], inp2[1], inp2[5])
            total_validation_loss += validation_loss
            '''
            print('Epoch {} Batch {} Loss {}'.format(str(epoch + 1),
                                                 str(batch),
                                                 str(batch_loss)))
            print('Epoch {} Batch {} Accu {}'.format(str(epoch + 1),
                                                 str(batch),
                                                 str(accuracy)))
            '''
            print('Epoch {} Batch {} Val_Loss {}'.format(str(epoch + 1),
                                                 str(batch),
                                                 str(validation_loss)))
            print('Epoch {} Batch {} Val_Accu {}'.format(str(epoch + 1),
                                                 str(batch),
                                                 str(validation_accuracy)))
        # saving (checkpoint) the model every epoch
        checkpoint.save(file_prefix = checkpoint_prefix)

        print('Epoch {} Loss {:.4f}'.format(epoch + 1,total_loss / steps_per_epoch))
        print('Time taken for 1 epoch {} sec\n'.format(time.time() - start))


#TESTING CASE:
else:
    #EVALUATE - INFERENCE
    #Following the paper, we perform the evaluation task without teacher forcing
    print("Starting testing")
    oov = False #wheter test the regular or OOV version of the testing dataset
    #We still used the same dataset object created at the beginning of the training
    #process: we need a starting training vocabulary.
    dataset.create_preprocessed_dataset(mode="testing", oov=oov)
    test_dataset, steps_per_epoch, vocab_dim, oov_max_length = create_dataset(dataset, task, "test", oov)
    decoder.oov_max_length = oov_max_length
    checkpoint.restore(tf.train.latest_checkpoint(checkpoint_dir))

    test_iterator = iter(test_dataset)
    average_accuracy = 0.0
    for batch in range(steps_per_epoch):
        inp = next(test_iterator)
        batch_accuracy, answer = test_step(inp[0], inp[3], inp[1], inp[5])
        average_accuracy += batch_accuracy
        if (batch %  10 == 0 or batch == steps_per_epoch-1):
            acc = average_accuracy / (batch+1)
            print("Batch " + str(batch) + ": Aver Accuracy: " + str(acc))
            print("Batch " + str(batch) + ": Current Accuracy: " + str(batch_accuracy))
            tf.print(tf.squeeze(answer), summarize=-1)
            #print_answer(answer)
            print("\n")
