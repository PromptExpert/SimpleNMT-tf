import tensorflow as tf
import argparse
import os
import pickle
from Config import Config
import time
import sys
import opts
from layers import *
import numpy as np

os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2' #解决Warning: The TensorFlow library wasn't compiled to use SSE4.2 instructions
parser = argparse.ArgumentParser()
opts.add_args(parser)
args = parser.parse_args()


def train():
    if args.checkpoint:
        saver.restore(sess,args.checkpoint)
    else:
        sess.run(tf.global_variables_initializer())
    for i in range(args.epochs):
        sess.run(iterator.initializer)
        total_loss = 0
        n_batches = 0
        try:
            while True:
                try:
                    _, loss_ = sess.run([update_step, loss])
                except tf.errors.InvalidArgumentError:
                    break
                print (loss_)
                total_loss += loss_
                n_batches += 1
        except tf.errors.OutOfRangeError:
            pass
        print('Average loss epoch {0}: {1:.2f}'.format(i, total_loss/n_batches))
        saver.save(sess, args.models_dir+"{0}th_epoch_model_{1:.2f}.ckpt".format(i,total_loss/n_batches))

def translate():
    saver.restore(sess,args.checkpoint)
    sess.run(iterator.initializer)
    src_char2index = pickle.load(open('preprocess/src_char2index.pickle', 'rb'))
    src_index2char = {v: k for k, v in src_char2index.items()}
    tgt_char2index = pickle.load(open('preprocess/tgt_char2index.pickle', 'rb'))
    tgt_index2char = {v: k for k, v in tgt_char2index.items()}
    try:
        while True:
            src,_translations = sess.run([src_batch,translations])
            src_chars = np.vectorize(src_index2char.get)(src)
            tgt_chars = np.vectorize(tgt_index2char.get)(_translations)
            for s,t in zip(src_chars,tgt_chars):
                print ('原文：{0}'.format(''.join(list(filter(lambda x: x != 'pad', s)))))
                print ('译文：{0}'.format(''.join(list(filter(lambda x: x != 'pad' and x != 'eos', t)))))
                print ()
    except tf.errors.OutOfRangeError:
        pass



if __name__ == "__main__":
    ########### Load Config ###########
    config_preprocess = pickle.load(open('preprocess/config_preprocess.pickle', 'rb'))
    config  = Config(args,config_preprocess)

    ########### Load Data ###########
    if args.tiny:
        filename = 'preprocess/tiny_preprocessed.pickle'
    elif args.translate:
        filename = 'preprocess/test_preprocessed.pickle'
    else:
        filename = 'preprocess/train_preprocessed.pickle'


    data = pickle.load(open(filename,'rb'))
    data = tf.data.Dataset.from_tensor_slices(data)
    data = data.shuffle(10000)
    data = data.batch(args.batch_size)
    iterator = data.make_initializable_iterator()
    src_batch,tgt_batch,shifted_tgt = iterator.get_next()

    ########### Define the inference model ###########
    #embedding
    embedding_encoder = tf.get_variable('src_embeddinng', [config.src_vocab_size, config.embedding_size],dtype = tf.float32)
    embedding_decoder = tf.get_variable('tgt_embeddinng', [config.tgt_vocab_size, config.embedding_size],dtype = tf.float32)

    encoder_inputs = tf.matrix_transpose(src_batch)  # make it time major
    decoder_inputs = tf.matrix_transpose(tgt_batch)  # make it time major

    encoder_emb_inputs = tf.nn.embedding_lookup(embedding_encoder,encoder_inputs)
    decoder_emb_inputs = tf.nn.embedding_lookup(embedding_decoder,decoder_inputs)

    #encoder
    #encoder = Encoder(config.rnn_size)
    encoder = biRNNEncoder(config.rnn_size)
    src_lengths = lengths(src_batch)
    #_,encoder_state = encoder(encoder_emb_inputs,src_lengths)
    #annotations = encoder(encoder_emb_inputs,src_lengths) #(20,32,1200), (src_sent_length,batch_size,rnn_size*2)
    #annotations,encoder_state = tf.transpose(encoder(encoder_emb_inputs,src_lengths), [1, 0, 2]) #(32,20,1200), (batch_size,src_sent_length,rnn_size*2)
    encoder_outputs,encoder_state = encoder(encoder_emb_inputs,src_lengths) #(32,20,1200), (batch_size,src_sent_length,rnn_size*2)
    attention_states = tf.transpose(encoder_outputs, [1, 0, 2])


    #attention
    attention_mechanism = tf.contrib.seq2seq.LuongAttention(args.attn_dim,attention_states,memory_sequence_length=src_lengths)
    #encoder_state = tf.contrib.seq2seq.AttentionWrapperState(encoder_state,attention_mechanism,0,None,False,attention_state = attention_states)
    #decoder
    tgt_lengths = lengths(tgt_batch)
    decoder = Decoder(args.attn_dim,
                      decoder_emb_inputs,
                      tgt_lengths,
                      config.tgt_vocab_size,
                      embedding_decoder,
                      args.batch_size,
                      attention_mechanism,
                      args.attn_dim)
    outputs = decoder(args.translate,encoder_state)

    ########### Define loss function ###########
    if not args.translate:
        decoder_outputs = tf.matrix_transpose(shifted_tgt)
        decoder_outputs = decoder_outputs[:tf.reduce_max(tgt_lengths)]
        target_weights = tf.cast(tf.greater(decoder_outputs,tf.zeros_like(decoder_outputs)),tf.float32)
        logits = outputs.rnn_output #shape (?,32,2748),(max_length,batch_size,tgt_vocab_size),max_length is the max of tgt_lengths
        cross_entropy = tf.nn.sparse_softmax_cross_entropy_with_logits(labels=decoder_outputs, logits=logits)
        loss = tf.reduce_sum(cross_entropy * target_weights/config.batch_size)

    ########### Define Optimizer ###########
        # Calculate and clip gradients
        params = tf.trainable_variables()
        gradients = tf.gradients(loss, params)
        clipped_gradients, _ = tf.clip_by_global_norm(gradients, config.max_gradient_norm)
        optimizer = tf.train.AdamOptimizer(args.learning_rate)
        update_step = optimizer.apply_gradients(zip(clipped_gradients, params))
    else:
        translations = tf.matrix_transpose(outputs.sample_id ) # make it batch major

    ############ Create Saver ###########
    saver = tf.train.Saver()

    ########### Training ###########
    with tf.Session() as sess:
        '''
        sess.run(tf.global_variables_initializer())
        sess.run(iterator.initializer)
        r1 = sess.run(annotations)
        print (r1.shape)
        #print (r2)
        sys.exit()
        '''

        if args.translate:
            translate()
        else:
            train()
