'''
src转id的时候不需要sos，tgt需要。eos都不需要。
'''
import argparse
import re
import sys
from functools import reduce
import pickle


def padding(sentence,max_length_sent):
    '''
    input is the index form of a sentence
    [int] -> [int]
    '''
    padded_sentence = sentence[:max_length_sent] if len(sentence) >= max_length_sent else sentence + [0] * (max_length_sent - len(sentence))
    return padded_sentence

def make_dict(sents,side):
    '''
    side is 'src' or 'tgt'
    '''
    chars = set([])
    for s in sents:
        chars = chars | set(s)

    indices = range(4,len(chars)+4)
    char2index = {char: cid for char, cid in zip(chars,indices)}
    char2index['sos'] = 3
    char2index['eos'] = 2
    char2index['unk'] = 1
    char2index['pad'] = 0

    pickle.dump(char2index,open(side + '_char2index.pickle','wb'))
    return char2index

def convert2index(sents,char2index,sent_length,tgt):
    '''
    [[char]] -> [[int]]
    '''
    result = []
    for s in sents:
        idx = [char2index[c] if c in char2index else 1 for c in s]
        if tgt:
            idx = [3] + idx
        result.append(padding(idx,sent_length))
    return result

def shift(s,length):
    '''
    shift a list to the left by one time step with an end-of-sentence tag appended on the right.
    '''
    a = s[1:length] + [2]
    zeros = s[length:]
    return a + zeros

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('-src_sent_length', type=int, default = 20,help="""the max length of source sentence""")
    parser.add_argument('-tgt_sent_length', type=int, default = 30,help="""the length of sentences truncated""")
    parser.add_argument('-train_prop', type=float, default = 0.95)
    args = parser.parse_args()

    #读取数据
    with open('data/synthetic_parallel_cls_10191.txt','r') as src:
        src_sents =[ s.split() for s in list(src) ] #type: [[char]]
    with open('data/synthetic_parallel_mdn_10191.txt','r') as mdn:
        tgt_sents =[ s.split() for s in list(mdn) ]

    #制作char2index字典，并存储为pickle。
    src_char2index = make_dict(src_sents,'src')
    tgt_char2index = make_dict(tgt_sents,'tgt')

    #将字变成index, 然后pad,存储
    src_indices = convert2index(src_sents,src_char2index,args.src_sent_length,False)
    lengths = [len(s) for s in tgt_sents]
    tgt_indices = convert2index(tgt_sents,tgt_char2index,args.tgt_sent_length,True)
    shifted_tgt = [shift(s,length) for s,length in zip(tgt_indices,lengths)]


    train_size = int(len(src_indices)*args.train_prop)
    pickle.dump((src_indices[:train_size],tgt_indices[:train_size],shifted_tgt[:train_size]),open('train_preprocessed.pickle','wb'))
    pickle.dump((src_indices[train_size:],tgt_indices[train_size:],shifted_tgt[train_size:]),open('test_preprocessed.pickle','wb'))
    pickle.dump((src_indices[:200],tgt_indices[:200],shifted_tgt[:200]),open('tiny_preprocessed.pickle','wb')) #小规模数据用于修改模型代码时测试可行性


    ############## 存储参数，用于训练 ##########
    config = {}
    config['src_vocab_size'] = len(src_char2index)
    config['tgt_vocab_size'] = len(tgt_char2index)
    config['src_sent_length'] = args.src_sent_length
    config['tgt_sent_length'] = args.tgt_sent_length
    pickle.dump(config,open('config_preprocess.pickle','wb'))
