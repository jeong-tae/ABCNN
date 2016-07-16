import cPickle
import numpy as np
import os
import sys
import io
from tqdm import tqdm

import pdb

def gloveLoad(glove_fpath, vocab):
    embed_path = glove_fpath + ".embed.pkl"
    word2vec_path = glove_fpath + ".word2vec.pkl"

    embed = None
    word2vec = {}
    flag = None

    if os.path.exists(word2vec_path):
        embed = load_pkl(embed_path)
        word2vec = load_pkl(word2vec)
        flag = True
    else:
        with io.open(glove_fpath + "/MCT_vectors2.txt", 'r') as f:
            for idx, l in tqdm(enumerate(f), "Load glove"):
                tokens = l.split(' ')
                word = tokens[0]
                vecs = tokens[1:]

                if vocab.has_key(word):
                    word2vec[word] = np.array(vecs, dtype="float32")

    return embed, word2vec, flag

def add_oov(word2idx, word2vec):
    for word in word2idx.keys():
        if not word2vec.has_key(word) and word != 'PAD':
            word2vec[word] = np.random.uniform(-0.25, 0.25, 50) # edim = 50
        elif word == 'PAD':
            word2vec[word] = np.zeros((50), np.float32)
    return word2vec # get return or not, same result.

def make_embed(word2idx, word2vec):
    embed_dim = 50 # from GloVe config

    embed = np.zeros(shape=(len(word2idx), embed_dim), dtype='float32')
    for idx, word in enumerate(word2vec.keys()):
        embed[word2idx[word]] = word2vec[word]

    return embed

def save(glove_fpath, embed, word2vec):
    embed_path = glove_fpath + ".embed.pkl"
    word2vec_path = glove_fpath + ".word2vec.pkl"
    save_pkl(embed, embed_path)
    save_pkl(word2vec, word2vec_path)

def word2vec_load(glove_fpath, vocab):
    embed, word2vec, flag = gloveLoad(glove_fpath, vocab)
    if flag:
        return embed, word2vec
    add_oov(vocab, word2vec)
    embed = make_embed(vocab, word2vec)
    return embed, word2vec

class Progress(object):
    """
    Progress bar
    """

    def __init__(self, iterable, bar_length=50):
        self.iterable = iterable
        self.bar_length = bar_length
        self.total_length = len(iterable)
        self.start_time = time.time()
        self.count = 0

    def __iter__(self):
        for obj in self.iterable:
            yield obj
            self.count += 1
            percent = self.count / self.total_length
            print_length = int(percent * self.bar_length)
            progress = "=" * print_length + " " * (self.bar_length - print_length)
            elapsed_time = time.time() - self.start_time
            print_msg = "\r|%s| %.0f%% %.1fs" % (progress, percent * 100, elapsed_time)
            sys.stdout.write(print_msg)
            if self.count == self.total_length:
                sys.stdout.write("\r" + " " * len(print_msg) + "\r")
            sys.stdout.flush()  
    
