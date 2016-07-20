"""import required packages and datapath"""
import numpy as np

class ABCNN(object):
    def __init__(self, train_storys, test_storys, questions, dictionary):
        self.dictionary     = dictionary
        self.voca_size      = len(dictionary)
        self.max_epoch      = 500
        self.max_seq_l      = max(len(train_storys), len(test_storys))
        self.epi_size       = max(train_storys.shape[1], test_storys.shape[1])
        self.train_range    = np.array(range(train_storys.shape[2]))
        self.test_range     = np.array(range(test_storys.shape[2]))
        self.batch_size     = 15
        self.init_std       = 0.02

        
        # hyper-parameter mentioned in Table1.
        self.alpha          = 0.2
        self.l2_norm        = 0.0065
        self.filter_width1  = 2
        self.filter_width2  = 2
        self.d_1, self.d_2  = [90, 90]
        self.top_k          = [1, 3] # how many snippet will choose at attention.
        self.lr             = 0.05

        # word representation
        self.edim           = 50 # which is d-dimension in paper.

        auxiliary_class = ["how much", "how many", "how", "what", "who", "where", "which", "when", "whose", "why", "will", "other"]

        self.idx2word = {}
        for key, value in self.dictionary.items():
            self.idx2word[value] = key
