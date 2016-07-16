import random
import numpy as np
import tensorflow as tf
import math

from .util import word2vec_load, Progress


import time
import pdb

random.seed(1003)
np.random.seed(1003)

glove_fpath = "/home/jtlee/projects/ABCNN/data/glove"

class ABCNN_TE(object):
    def __init__(self, config, sess, train_data, test_data):

        self.vocab = config.dictionary
        self.embed, self.word2vec = word2vec_load(glove_fpath, self.vocab)

        self.sess = sess

        self.voca_size = config.voca_size
        self.max_words = config.max_seq_l
        self.epi_size = config.epi_size
        
        self.train_storys, self.train_questions, self.train_question_type, self.train_answer_candidates, self.train_answers = train_data
        self.test_storys, self.test_questions, self.test_question_type, self.test_answer_candidates, self.test_answers = test_data
        self.train_range = config.train_range
        self.test_range = config.test_range

        # training detail
        self.batch_size = config.batch_size
        self.edim = config.edim
        self.init_std = config.init_std
        #self.max_grad_norm = config.max_grad_norm
        self.max_epoch = config.max_epoch
        self.lr = tf.Variable(config.lr, trainable = False)
        self.filter_width1 = config.filter_width1
        self.filter_width2 = config.filter_width2
        self.d_1 = config.d_1
        self.d_2 = config.d_2
        self.alpha = config.alpha

        self.input_querys = tf.placeholder(tf.float32, [self.batch_size, self.max_words, self.edim])
        self.input_episodes = tf.placeholder(tf.float32, [self.batch_size, self.epi_size, self.max_words, self.edim])

        self.input_answers = tf.placeholder(tf.float32, [self.batch_size, 1])
        # 1 for answer index which range in (0, 4)
        self.input_candidates = tf.placeholder(tf.float32, [self.batch_size, 4, self.max_words, self.edim])

        self.keep_prob = tf.placeholder('float')

    def build_model(self):
        print(" [*] building...")
        def s_length(s_data): # not used
            used = tf.sign(tf.reduce_max(tf.abs(s_data), reduction_indices=2))
            length = tf.reduce_sum(used, reduction_indices=1)
            length = tf.cast(length, tf.int32)
            return length

        def d_length(d_data): # not used
            word_max = tf.sign(tf.reduce_max(tf.abs(d_data), reduction_indices=3))
            used = tf.sign(tf.reduce_max(tf.abs(word_max), reduction_indices=2))
            length = tf.reduce_sum(used, reduction_indices=1)
            length = tf.cast(length, tf.int32)
            return length
        with tf.variable_scope("Sentence_CNN"):
            self.W1 = tf.get_variable('CNN_weights', [self.d_1, self.filter_width1 * self.edim], initializer = tf.random_normal_initializer(0, stddev = self.init_std))
            self.b1 = tf.get_variable('CNN_biases', [self.d_1], stddev = self.init_std, initializer = tf.random_normal_initializer(0, stddev = self.init_std))

        def wide_CNN(self, data, w, l, scope):
            feature_map = []
            padded_data = tf.pad(data, [[0, 0], [math.floor(w/2), math.ceil(w/2)], [0, 0]], "CONSTANT")
            with tf.variable_scope(scope, reuse = True):
                weights = get_variable('CNN_weights')
                biases = get_variable('CNN_biases')
                for t in range(w-1, l+w):
                    c_i = tf.reshape(padded_data[:, t-w+1:t+1, :], [self.batch_size, -1])
                    p_i = tf.tanh(tf.matmul(c_i, tf.transpose(weights)) + biases)
                    p_i = tf.reshape(p_i, [self.batch_size, 1, -1])
                    feature_map.append(p_i)
            return tf.concat(1, feature_map)

            ##################### Sentence-CNN ######################
        sentence_CNN_q = wide_CNN(self.input_querys, self.filter_width1, self.max_words, 'Sentence_CNN')
        #sentence_CNN_a = sentence_CNN(self.input_answers, self.filter_width1, self.max_words)

        sentences_CNN_c = []
        for n in range(4):
            sentence = self.input_candidates[:, n, :, :]
            sentence_CNN_c = wide_CNN(sentence_in, self.filter_width1, self.max_words, 'Sentence_CNN')
            sentence_CNN_c = tf.reshape(sentence_CNN_c, [self.batch_size, 1, -1, self.d_1])
            sentences_CNN_c.append(sentence_CNN_c)
        sentences_CNN_c = tf.concat(1, sentences_CNN_c)

        sentences_CNN_d = []
        for n in range(self.epi_size):
            sentence = self.input_episodes[:, n, :, :]
            sentence_CNN_d = wide_CNN(sentence, self.filter_width1, self.max_words, 'Sentence_CNN')
            sentence_CNN_d = tf.reshape(sentence_CNN_d, [self.batch_size, 1, -1, self.d_1])
            sentences_CNN_d.append(sentence_CNN_d)
        sentences_CNN_d = tf.concat(1, sentences_CNN_d)
        #################### End of Sentence-CNN #################

        ################# Sentence-Level Representation ##############
        r_sq = tf.reduce_max(sentence_CNN_q, reduction_indices = 1)
        r_sc = []
        for n in range(4):
            p_i = sentences_CNN_c[:, n, :, :]
            elwise_maxpool = tf.reduce_max(p_i, reduction_indices = 1)
            elwise_maxpool = tf.reshape(elwise_maxpool, [self.batch_size, 1, -1])
            r_sc.append(elwise_maxpool)
        r_sc = tf.concat(1, r_sc)

        SLR_d = []
        for n in range(self.epi_size):
            p_i = sentences_CNN_d[:, n, :, :]
            elwise_maxpool = tf.reduce_max(p_i, reduction_indices = 1)
            elwise_maxpool = tf.reshape(elwise_maxpool, [self.batch_size, 1, -1])
            SLR_d.append(elwise_maxpool)
        SLR_d = tf.concat(1, SLR_d)

        sq_attention = tf.batch_matmul(tf.nn.l2_normalize(tf.reshape(r_sq, [self.batch_size, 1, -1]), dim = 2), tf.nn.l2_normalize(SLR_d, dim = 2), adj_y = True)
        sc_attention = tf.batch_matmul(tf.nn.l2_normalize(r_sc, dim =2), tf.nn.l2_normalize(SLR_d, dim = 2), adj_y = True)

        top_sq = tf.nn.top_k(tf.reshape(sq_attention, [self.batch_size, -1]), k = 1)[1]
        top_sc = tf.nn.top_k(sc_attention, k = 1)[1]

        def index2d_to_2dtensor(self, index2d, batch_size, length):
            indices = tf.reshape(range(0, batch_size, 1), [batch_size, 1])
            concatenated = tf.concat(1, [indices, index2d])
            concat = tf.concat(0, [[batch_size], [length]])
            output_shape = tf.reshape(concat, [2])
            sparse_to_dense = tf.sparse_to_dense(concatenated, output_shape, 1.0, 0.0)
            return tf.reshape(sparse_to_dense, [batch_size, length])

        sq_dense = index2d_to_2dtensor(top_sq, self.batch_size, self.epi_size)
        sq_dense = tf.reshape(sq_dense, [self.batch_size, self.epi_size, 1])
        masked_sq = tf.mul(SLR_d, sq_dense)
        v_sq = tf.reduce_max(masked_sq, reduction_indices = 1)

        v_sc = [] # list of 4 elements
        for n in range(4):
            sc_dense = index2d_to_2dtensor(top_sc[:, n, :], self.batch_size, self.epi_size)
            sc_dense = tf.reshape(sc_dense, [self.batch_size, self.epi_size, 1])
            masked_sc = tf.mul(SLR_d, sc_dense)
            v_sc.append(tf.reshape(tf.reduce_max(masked_sc, reduction_indices = 1), [self.batch_size, 1, -1]))
        v_sc = tf.concat(1, v_sc)            
        ############## End of Sentence-Level Representation #############

        ########################## Snippet CNN ##########################
        with tf.variable_scope("Snippet_CNN"):
            self.W2 = tf.get_variable('CNN_weights', [self.d_2, self.filter_width2 * self.d_1], initializer = tf.random_normal_initializer(0, stddev = self.init_std))
            self.b2 = tf.get_variable('CNN_biases', [self.d_1], stddev = self.init_std, initializer = tf.random_normal_initializer(0, stddev = self.init_std))

        snippet_CNN_q = wide_CNN(tf.reshape(r_sq, [self.batch_size, 1, -1]), self.filter_width2, 1, 'Snippet_CNN')
        snippets_CNN_c = []
        for n in range(4):
            temp = tf.reshape(r_sc[:, n, :], [self.batch_size, 1, -1])
            snippet_CNN_c = wide_CNN(temp, self.filter_width2, 1, 'Snippet_CNN')
            snippet_CNN_c = tf.reshape(snippet_CNN_c, [self.batch_size, 1, -1, self.d_2])
            snippets_CNN_c.append(snippet_CNN_c)
        snippets_CNN_c = tf.concat(1, snippets_CNN_c)
        snippet_CNN_d = wide_CNN(SLR_d, self.filter_width2, self.epi_size, 'Snippet_CNN')
        ####################### End of Snippet CNN ######################

        ################# Snippet-Level Representation ##################
        r_tq = tf.reduce_max(snippet_CNN_q, reduction_indices = 1)
        r_tc = []
        for n in range(4):
            p_i = snippets_CNN_c[:, n, :, :]
            elwise_maxpool = tf.reduce_max(p_i, reduction_indices = 1)
            elwise_maxpool = tf.reshape(elwise_maxpool, [self.batch_size, 1, -1])
            r_tc.append(elwise_maxpool)
        r_tc = tf.concat(1, r_tc)
        TLR_d = snippet_CNN_d

        tq_attention = tf.batch_matmul(tf.nn.l2_normalize(tf.reshape(r_tq, [self.batch_size, 1, -1]), dim = 2), tf.nn.l2_normalize(SLR_d, dim = 2), adj_y = True)
        tc_attention = tf.batch_matmul(tf.nn.l2_normalize(r_tc, dim = 2), tf.nn.l2_normalize(TLR_d, dim = 2), adj_y = True)

        top_tq = tf.nn.top_k(tf.reshape(tq_attention, [self.batch_size, -1]), k = 3)[1]
        top_tc = tf.nn.top_k(tc_attention, k = 3)[1]

        tq_dense = index2d_to_2dtensor(top_tq, self.batch_size, self.epi_size)
        tq_dense = tf.reshape(tq_dense, [self.batch_size, self.epi_size, 1])
        masked_tq = tf.mul(TLR_d, tq_dense)
        v_tq = tf.reduce_max(masked_tq, reduction_indices = 1)

        v_tc = [] # list of 4 elements
        for n in range(4):
            tc_dense = index2d_to_2dtensor(top_tc[:, n, :], self.batch_size, self.epi_size)
            tc_dense = tf.reshape(tc_dense, [self.batch_size, self.epi_size, 1])
            masked_tc = tf.mul(SLR_d, tc_dense)
            v_tc.append(tf.reshape(tf.reduce_max(masked_tc, reduction_indices = 1), [self.batch_size, 1, -1]))
        v_tc = tf.concat(1, v_tc)
        ################ End of Snippet-Level Representation #############

        ####################### Overall Representation ###################
        with tf.variable_scope("Highway"):
            self.W3 = tf.get_variable('weights', [self.d_1, self.d_1], initializer = tf.random_normal_initializer(0, stddev = self.init_std))
            self.b3 = tf.get_variable('biases', [self.d_1], stddev = self.init_std, initializer = tf.random_normal_initializer(0, stddev = self.init_std))
        
        def highway_layer(self, _s, _t, scope):
            with tf.variable_scope(scope, reuse = True):
                weights = get_variable('weights')
                biases = get_variable('biases')
                h = tf.sigmoid(tf.matmul(_s, weights) + biases)
                v_o = tf.mul((1-h), _s) + tf.mul(h, _t)
            return v_o

        v_oq = highway_layer(v_sq, v_tq, 'Highway')
        r_iq = highway_layer(r_sq, r_tq, 'Highway')

        v_oc = []
        for n in range(4):
            sc = v_sc[:, n, :]
            tc = v_tc[:, n, :]
            oc = highway_layer(sc, tc, 'Highway')
            v_oc.append(oc)
        v_oc = tf.concat(1, v_oc)
            
        r_ic = []
        for n in range(4):
            sc = r_sc[:, n, :]
            tc = r_tc[:, n, :]
            oc = highway_layer(sc, tc, 'Highway')
            r_ic.append(oc)
        r_ic = tf.concat(1, r_ic)

        ############ HABCNN-TE ##########
        attention = tf.batch_matmul(tf.nn.l2_normalize(tf.reshape(v_oq, [self.batch_size, 1, -1]), dim = 2), tf.nn.l2_normalize(r_ic, dim = 2), adj_y = True)

        self.preds = tf.nn.top_k(tf.reshape(attention, [self.batch_size, -1]), k = 1)[1]
            
        cost = self.alpha
            
        partitions = tf.one_hot(self.input_answers, 4, 1, 0)
        pos, neg = tf.dynamic_partition(r_ic, partitions, 2)
        pos = tf.reshape(pos, [self.batch_size, 1, -1])
        neg = tf.reshape(neg, [self.batch_szie, 3, -1])

        neg_cost = tf.batch_matmul(tf.nn.l2_normalize(tf.reshape(v_oq, [self.batch_size, 1, -1]), dim = 2), tf.nn.l2_normalize(neg, dim = 2), adj_y = True)        
        pos_cost = tf.batch_matmul(tf.nn.l2_normalize(tf.reshape(v_oq, [self.batch_size, 1, -1]), dim = 2), tf.nn.l2_normalize(pos, dim = 2), adj_y = True)

        cost = cost + neg_cost - pos_cost
        self.loss = tf.max(0, tf.reduce_mean(cost))

        self.opt = tf.train.AdamOptimizer(self.lr).minimize(loss)
        print(" [*] Bulid done.")

    def train():
        # init
        total_preds = []
        total_answers = []
            
        for step, (querys, episodes, candidates, answers) in Progress(enumerate(self.data_iteration(self.train_storys, self.train_questions, self.train_answer_candidates, self.train_answers, True))):
            _, loss, preds = self.sess.run([self.opt, self.loss, self.preds], feed_dict = { self.input_querys: querys,
            self.input_episodes: episodes,
            self.input_candidates: candidates,
            self.input_answers: answers
            })
            total_preds.append(preds)
            total_answers.append(answers)
        total_preds = np.concatenate(total_preds, axis = 0)
        total_answers = np.concatenate(total_answers, axis = 0)
        return loss, self.accuracy(total_preds, total_answers)   

    def test():
        total_preds = []
        total_answers = []

        for step, (querys, episodes, candidates, answers) in enumerate(self.data_iteration(self.test_storys, self.test_questions, self.test_answer_candidates, self.test_answers, True)):
            loss, preds = self.sess.run([self.loss, self.preds], feed_dict = { self.input_querys: querys,
            self.input_episodes: episodes,
            self.input_candidates: candidates,
            self.input_answers: answers
            })
            total_preds.append(preds)
            total_answers.append(answers)
        total_preds = np.concatenate(total_preds, axis = 0)
        total_answers = np.concatenate(total_answers, axis = 0)
        return loss, self.accuracy(total_preds, total_answers)

    def run(self, task_id):
        self.task_id = task_id
        tf.initialize_all_variables().run()

        for i in range(self.max_epoch):
            train_loss, train_acc = self.train()
            test_loss, test_acc = self.test()
            print("Epoch: %d, Train loss: %.3f, Train Acc: %.3f" % (i+1, train_loss, train_acc))
            print("Epoch: %d, Test loss: %.3f, Test Acc: %.3f" % (i+1, test_loss, test_acc))
            
        test_loss, test_acc = self.test()
        print("Task: %d, Test loss: %.3f, Test Acc: %.3f" % (task_id, test_loss, test_acc))

    def data_iteration(self, story, questions, answer_candidates, answers, is_train = True):
        data_range = None
        if is_train:
            random.shuffle(self.train_range)
            data_range = self.train_range
        else:
            data_range = self.test_range
        batch_len = len(data_range) // self.batch_size

        for l in xrange(batch_len):
            sbatch = data_range[self.batch_size * l:self.batch_size * (l+1)]
            qbatch = np.random.randint(4, size = self.batch_size)

            batch_querys = np.zeros((self.batch_size, self.max_words, self.edim), np.float32)
            batch_episodes = np.zeros((self.batch_size, self.epi_size, self.max_words, self.edim), np.float32)
            batch_candidates = np.zeros((self.batch_size, 4, self.max_words, self.edim), np.float32)
            batch_answers = np.zeros((self.batch_size, 1), np.int32)

            for b in range(self.batch_size):
                episode = np.copy(story[:, :, sbatch[b]])
                query = np.copy(questions[:, qbatch[b], sbatch[b]])
                candidates = np.copy(answer_candidates[:, :, qbatch[b], sbatch[b]])
                answer = np.copy(answers[qbatch[b], sbatch[b]])

                for i in range(episode.shape[1]):
                    sentence = episode[:, i]
                    for j in range(sentence.shape[0]):
                        batch_episodes[b, i, j, :] = self.embed[sentence[j]]

                for j in range(query.shape[0]):
                    batch_querys[b, j, :] = self.embed[query[j]]

                for i in range(candidates.shape[1]):
                    candidate = candidates[i]
                    for j in range(candidate.shape[0]):
                        batch_candidates[b, i, j, :] = self.embed[candidate[j]]
                batch_answers[b, :] = answer

            yield (batch_querys, batch_episodes, batch_candidates, batch_answers)

    def accuracy(self, predictions, labels):
        return (100.0 * np.sum(predictions == labels) / predictions.shape[0])












