def add_args(parser):
    parser.add_argument('-embedding_size', type=int, default = 2000)
    parser.add_argument('-rnn_size', type=int, default = 600,help="""the number of units of encoder""")
    parser.add_argument('-attn_dim', type=int, default = 500,help="""the number of units of attention""")
    #parser.add_argument('-rnn_size_sent', type=int, default = 300,help="""the number of units of sentence encoder""")
    parser.add_argument('-batch_size', type=int, default = 32)
    parser.add_argument('-learning_rate', type=float, default = 0.002)
    parser.add_argument('-max_gradient_norm', type=float, default = 4.0)
    parser.add_argument('-epochs', type=int, default = 30)
    parser.add_argument('-checkpoint', type=str,default = None)
    parser.add_argument('-test', action="store_true")
    parser.add_argument('-models_dir',type=str, default = 'models/',help="""the directory to save checkpoints""")
    parser.add_argument('-tiny',action='store_true')
    parser.add_argument('-translate',action = 'store_true')