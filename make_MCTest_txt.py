#################################################
# To use GloVe, make txt corpus as a input file #
# All words separated by a single space         #
# ###############################################

import os
import pandas as pd
import nltk
from tqdm import tqdm

COLUMNS = ['id', 'description', 'story',
        'q1', 'a11', 'a12', 'a13', 'a14',
        'q2', 'a21', 'a22', 'a23', 'a24',
        'q3', 'a31', 'a32', 'a33', 'a34',
        'q4', 'a41', 'a42', 'a43', 'a44']

QUESTION_TYPES = ['one', 'multiple']
PUNCTS = ['.', '?', ',', '!', '"', '\'', '$', '%', '^', '&']

rm_stop     = True
rm_punct    = True

import pdb

def question_text(question):
    return question.split(':')[1].strip()

def question_type(question):
    question_type, _ = question.split(':')
    assert question_type in QUESTION_TYPES
    return question_type

def tokenize(token_mappers, text):
    if not isinstance(text, basestring):
        text = str(text)
    text = text.replace('\\newline', ' ')
    mapped = nltk.word_tokenize(text)
    for mapper in token_mappers:
        mapped = filter(lambda x: x is not None, map(mapper, mapped))
    return mapped

def datum_to_token(datum, include_answers = True):
    #pdb.set_trace()
    tokens = datum['story']
    for question in datum['mcQuestion']:
        tokens.extend(question['question'])
        if include_answers:
            for answer in question['answers']:
                tokens.extend(answer)
    return ' '.join(map(lambda t: t.lower(), tokens))

def tokenization(data_files = None, stop_fpath = None, fout_path = 'MCTest.txt'):

    for path in data_files:
        assert path != None and os.path.exists(path), "Something wrong with path: '" + path +"'"    
    token_mappers = []

    if stop_fpath == None:
        print(" [*] Stopwords are not used...")
    
    if rm_stop and stop_fpath:
        stopwords = open(stop_fpath, 'r').read().split('\n')
        stopwords = set(map(lambda x: x.strip().rstrip(), stopwords))
        token_mappers.append(lambda x: x if x.lower() not in stopwords else None)

    if rm_punct:
        token_mappers.append(lambda x: x if x not in PUNCTS else None)

    data_path, ans_path = data_files
    
    story_questions_in = open(data_path)
    story_questions = pd.read_csv(story_questions_in, sep = '\t', names=COLUMNS)
    story_questions_in.close()

    fout = open(fout_path, 'w')

    for story_question in tqdm([story_questions.ix[i] for i in story_questions.index]):
        datum = {   'id': story_question['id'],
                    'story': tokenize(token_mappers, story_question['story']),
                    'mcQuestion' : [{
                        'question': tokenize(token_mappers, question_text(story_question['q%d' % q_number])),
                        'answers': [tokenize(token_mappers, story_question['a%d%d' % (q_number, a_number)]) for a_number in xrange(1, 5)],
                        'type': question_type(story_question['q%d' % q_number])
                        } for q_number in xrange(1, 5)]
                }

        fout.write(datum_to_token(datum) + ' ')
    fout.close()


if __name__ == '__main__':
    data_dir = "/home/jtlee/projects/ABCNN/data/MCTest"
    test_ans_dir = "/home/jtlee/projects/ABCNN/data/MCTestAnswers"
    task_id = 500

    train_files = ['%s/mc%d.train.tsv' % (data_dir, task_id), '%s/mc%d.train.ans' % (data_dir, task_id)]

    test_files = ['%s/mc%d.test.tsv' % (data_dir, task_id), '%s/mc%d.test.ans' % (test_ans_dir, task_id)]

    print " [*]Start tokenization..."
    tokenization(train_files, '/home/jtlee/projects/ABCNN/stopword_mini.txt', 'MCTest2.txt')

    print " [*]done!"




