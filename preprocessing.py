import sys
import os
import pandas as pd
import nltk # it is for word tokenizing.
import cPickle
import numpy as np # ...
from tqdm import tqdm
import re

import pdb

############################################
# preprocessing utility is defined in here #
# ##########################################

DATA_TYPE = 'TE' # or BASE
COLUMNS = ['id', 'description', 'story',
        'q1', 'a11', 'a12', 'a13', 'a14',
        'q2', 'a21', 'a22', 'a23', 'a24',
        'q3', 'a31', 'a32', 'a33', 'a34',
        'q4', 'a41', 'a42', 'a43', 'a44']

QUESTION_TYPES = ['one', 'multiple']
PUNCTS = ['.', '?', ',', '!', '"', '\'', '$', '%', '^', '&']

rm_stop     = False
rm_punct    = True

def question_text(question):
    return question.split(':')[1].strip()

def question_type(question):
    question_type, _ = question.split(':')
    assert question_type in QUESTION_TYPES
    return question_type

def gloveLoad(glove_fpath, vocab):
    embed_path = glove_fpath + ".embed.pkl"
    word2vec_path = glove_fpath + ".word2vec.pkl"

    embed = None
    word2vec = None

    if os.path.exists(word2vec_path):
        embed = load_pkl(embed_path)
        word2vec = load_pkl(word2vec)
    else:
        with io.open(glove_fpath, 'r') as f:
            for idx, l in tqdm(enumerate(f), "Load glove"):
                tokens = l.split(' ')
                word = tokens[0]
                vecs = tokens[1:]

                if vocab.has_key(word):
                    word2vec[word] = np.array(vecs, dtype="float32")

    return embed, word2vec

def make_embed(word2idx, word2vec, vocab):
    embed_dim = 50 # from GloVe config

    embed = np.zeros(shape=(vocab.size, embed_dim), dtype='float32')
    for idx, word in enumerate(word2vec.keys()):
        embed[word2idx[word]] = word2vec[word]

    return embed

def sentence_split(text):
    splited = re.split(r'[\.|!|\?]', text)
    return [ s.strip() for s in splited ]

def tokenize(token_mappers, text):
    if not isinstance(text, basestring):
        text = str(text)
    text = text.replace('\\newline', ' ')
    mapped = nltk.word_tokenize(text)
    for mapper in token_mappers:
        mapped = filter(lambda x: x is not None, map(mapper, mapped))
    return mapped

def mctest_load(data_files = None, stop_fpath = None, dictionary = None):
 
    """ Parse MCTest data.
        Args:
            data_files: consists of QA file and Answer file.
            data_files = [Multi-choice QA, Answer file]
            stop_fpath: stopword file path.
            dictionary:
        Returns:
            storys:
            questions:
            question_types:
            answer_candidatas:
            answers:
    """
       
    for path in data_files:
        assert path != None and os.path.exists(path), "Something wrong with path: '" + path + "'"

    token_mappers = []
    
    if stop_fpath == None:
        print(" [*] Stopwords are not used...")

    if rm_stop and stop_fpath:
        stopwords = open(stop_fpath, 'r').read().split('\n')
        stopwords = set(map(lambda x: x.strip().rstrip(), stopword))
        token_mappers.append(lambda x: x if x.lower() not in stopwords else None)

    if rm_punct:
        token_mappers.append(lambda x: x if x not in PUNCTS else None)

    storys              = np.zeros((100, 100, 500), np.int16)
    questions           = np.zeros((100, 4, 500), np.int16)
    question_types      = np.zeros((4, 500), np.int16)
    answer_candidates   = np.zeros((100, 4, 4, 500), np.int16)
    answers             = np.zeros((4, 500), np.int16)

    story_idx, max_words, max_sentences = -1, 0, 0

    data_path, ans_path = data_files

    story_questions_in = open(data_path)
    story_questions = pd.read_csv(story_questions_in, sep='\t', names=COLUMNS)
    story_questions_in.close()

    for story_question in (story_questions.ix[i] for i in story_questions.index):
        _id = story_question['id']
        _story = sentence_split(story_question['story'])
        story_idx += 1
        for i in range(len(_story)):
            tokens = tokenize(token_mappers, _story[i])
            for j in range(len(tokens)):
                w = tokens[j]
                if w not in dictionary:
                    dictionary[w] = len(dictionary)
                if max_words < j:
                    max_words = j
                storys[j, i, story_idx] = dictionary[w]

        for q_number in range(4):
            _question = tokenize(token_mappers, question_text(story_question['q%d' % (q_number+1)]))
            for j in range(len(_question)):
                w = _question[j] # check ? mark is in w. after check del this.
                if w not in dictionary:
                    dictionary[w] = len(dictionary)
                questions[j, q_number, story_idx] = dictionary[w]
            _type = question_type(story_question['q%d' % (q_number+1)])
            if _type == "multiple":
                question_types[q_number, story_idx] = 1            

            for a_number in range(4):
                _answer = tokenize(token_mappers, story_question['a%d%d' % (q_number+1, a_number+1)])
                for k in range(len(_answer)):
                    w = _answer[k]
                    if max_words < k:
                        max_words = k
                    if w not in dictionary:
                        dictionary[w] = len(dictionary)
                    answer_candidates[k, a_number, q_number, story_idx] = dictionary[w]

    ans_in = open(ans_path).readlines()
    for idx, ans_line in enumerate(ans_in):
        ABCD = ans_line.split()
        for j in range(len(ABCD)):
            ans_idx = ABCD[j]
            if 'A' == ABCD[j]:
                ans_idx = 0
            elif 'B' == ABCD[j]:
                ans_idx = 1
            elif 'C' == ABCD[j]:
                ans_idx = 2
            else:
                ans_idx = 3
            answers[j, idx] = ans_idx

    storys          = storys[:max_words+1, :max_sentences, :(story_idx+1)]
    questions       = questions[:max_words+1, :4, :(story_idx+1)]
    question_types  = question_types[:4, :(story_idx+1)]
    answer_candidates = answer_candidates[:max_words+1, :4, :4, :(story_idx+1)] 
    # max_words + 1 for <EOS>

    return storys, questions, question_types, answer_candidates, answers
