'''
sub-components of the hierachical attention network
'''
import tensorflow as tf

def lengths(batch):
    '''
    compute the actual lengths of each of the sequences in the batch
    the input shape is (batch, sent_length)
    return shape is (batch,)
    '''
    batch = tf.sign(batch)
    lengths = tf.reduce_sum(batch,1)
    return lengths


class biRNNEncoder(object):
    def __init__(self,rnn_size):
        self.encoder_cell_fw = tf.nn.rnn_cell.BasicLSTMCell(rnn_size)
        self.encoder_cell_bw = tf.nn.rnn_cell.BasicLSTMCell(rnn_size)

    def __call__(self,encoder_emb_inputs,lengths):
        bi_outputs, encoder_state = tf.nn.bidirectional_dynamic_rnn(self.encoder_cell_fw,
                                                                    self.encoder_cell_bw,
                                                                    encoder_emb_inputs,
                                                                    sequence_length=lengths,
                                                                    dtype=tf.float32,
                                                                    time_major=True)
        '''
        ((fw_outputs, bw_outputs), (output_state_fw, output_state_bw)) = tf.nn.bidirectional_dynamic_rnn(cell_fw=self.encoder_cell_fw,
                                                                             cell_bw=self.encoder_cell_bw,
                                                                             inputs=encoder_emb_inputs,
                                                                             dtype=tf.float32,
                                                                             time_major=True,
                                                                             sequence_length = lengths)
        '''
        encoder_outputs = tf.concat(bi_outputs, -1)
        #annotations = tf.concat((fw_outputs, bw_outputs), 2)
        #encoder_state = tf.concat((output_state_fw, output_state_bw), 2)
        #return annotations,encoder_state
        return encoder_outputs,encoder_state


class Encoder(object):
    def __init__(self,rnn_size):
        self.encoder_cell = tf.nn.rnn_cell.BasicLSTMCell(rnn_size)

    def __call__(self,encoder_emb_inputs,lengths):
        encoder_outputs, encoder_state = tf.nn.dynamic_rnn(
                                                           self.encoder_cell,
                                                           encoder_emb_inputs,
                                                           sequence_length=lengths,
                                                           time_major=True,
                                                           dtype = tf.float32)
        return encoder_outputs,encoder_state

class Decoder(object):
    def __init__(self,rnn_size,decoder_emb_inputs,lengths,tgt_vocab_size,embedding_decoder,batch_size,attention,attn_dim):
        self.batch_size = batch_size
        self.decoder_cell = tf.nn.rnn_cell.BasicLSTMCell(rnn_size)
        self.decoder_cell = tf.contrib.seq2seq.AttentionWrapper(self.decoder_cell, attention, attention_layer_size=attn_dim)
        self.train_helper = tf.contrib.seq2seq.TrainingHelper(decoder_emb_inputs, lengths, time_major=True)
        self.projection_layer = tf.layers.Dense(tgt_vocab_size, use_bias=False)
        self.translate_helper = tf.contrib.seq2seq.GreedyEmbeddingHelper(embedding_decoder,tf.fill([batch_size], 3), 2)


    def __call__(self,translate,encoder_state):
        helper = self.translate_helper if translate else self.train_helper
        initial_state = self.decoder_cell.zero_state(dtype=tf.float32, batch_size=self.batch_size)
        decoder = tf.contrib.seq2seq.BasicDecoder(self.decoder_cell, helper, initial_state = initial_state,output_layer= self.projection_layer)
        outputs,_,_ = tf.contrib.seq2seq.dynamic_decode(decoder = decoder,
                                                        output_time_major = True,
                                                        impute_finished = True,
                                                        maximum_iterations = 30)
        return outputs
