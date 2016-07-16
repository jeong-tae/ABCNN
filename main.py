import numpy as np
import tensorflow as tf
import glob

from config import ABCNN as ABCNN_config
from preprocessing import mctest_load

from models import ABCNN_TE
import pdb

stop_fpath = "/home/jtlee/projects/ABCNN/stopword_mini.txt"
data_dir = "/home/jtlee/projects/ABCNN/data/MCTest"
test_ans_dir = "/home/jtlee/projects/ABCNN/data/MCTestAnswers"

def run_task(data_dir, test_ans_dir, stop_fpath, task_id):

    train_files = ['%s/mc%d.train.tsv' % (data_dir, task_id), '%s/mc%d.train.ans' % (data_dir, task_id)]

    test_files = ['%s/mc%d.test.tsv' % (data_dir, task_id), '%s/mc%d.test.ans' % (test_ans_dir, task_id)]

    dictionary = {"PAD" : 0}

    train_storys, train_questions, train_question_type, train_answer_candidates, train_answers = mctest_load(train_files, stop_fpath, dictionary)
    test_storys, test_questions, test_question_type, test_answer_candidates, test_answers = mctest_load(test_files, stop_fpath, dictionary)

    abcnn_config = ABCNN_config(train_storys, test_storys, train_questions, dictionary)

    with tf.Session() as sess:
        model = ABCNN_TE(abcnn_config, sess, (train_storys, train_questions, train_question_type, train_answer_candidates, train_answers), (test_storys, test_questions, test_question_type, test_answer_candidates, test_answers))
        model.build_model()
        model.run(task_id)

def main(_):
    
    run_task(data_dir, test_ans_dir, stop_fpath, 500)

if __name__ == '__main__':
    tf.app.run()



