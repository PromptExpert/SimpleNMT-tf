class Config(object):
    def __init__(self,args,config_preprocess):
        self._embedding_size = args.embedding_size
        self._rnn_size = args.rnn_size
        self._src_sent_length = config_preprocess['src_sent_length']
        self._tgt_sent_length = config_preprocess['tgt_sent_length']
        #self._sent_num = config_preprocess['sent_num']
        self._src_vocab_size = config_preprocess['src_vocab_size']
        self._tgt_vocab_size = config_preprocess['tgt_vocab_size']
        #self._num_labels = config_preprocess['num_labels']
        self._batch_size = args.batch_size
        self._max_gradient_norm = args.max_gradient_norm

    @property
    def embedding_size(self):
        return self._embedding_size

    @property
    def rnn_size(self):
        return self._rnn_size


    @property
    def src_sent_length(self):
        return self._src_sent_length

    @property
    def tgt_sent_length(self):
        return self._tgt_sent_length

    @property
    def src_vocab_size(self):
        return self._src_vocab_size

    @property
    def tgt_vocab_size(self):
        return self._tgt_vocab_size

    @property
    def batch_size(self):
        return self._batch_size

    @property
    def max_gradient_norm(self):
        return self._max_gradient_norm
