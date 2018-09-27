import ast
from .base_tokenizer import BaseTokenizer
from .utils import load_n_grams
import pycrfsuite
import sklearn_crfsuite
import os
import pickle
import string
__author__ = "Cao Bot"
__copyright__ = "Copyright 2018, DeepAI-Solutions"


def load_crf_config(file_path):
    """
    Load config from file. E.g. {'c1': 1.0, 'c2': 0.001, ...}
    Note: You can rewrite this function using json library
    :param file_path: path to config file
    :return: config as dictionary
    """
    with open(file_path) as fr:
        param_dict = fr.read()
        param_dict = ast.literal_eval(param_dict)
    return param_dict


def wrapper(func, args):
    """Call a function with argument list"""
    return func(*args)


def load_data_from_file(data_path):
    """
    Load data from file
    :param data_path: input file path
    :return: sentences and labels
    Examples
    sentences = [['Hello', 'World'], ['Hello', 'World']]
    labels = [['B_W', 'I_W'], ['B_W', 'I_W']]
    """
    sentences = []
    labels = []
    with open(data_path, 'r') as fr:
        lines = fr.readlines()
        for line in lines:
            sent = []
            sent_labels = []
            for word in line.strip().split():
                syllables = word.split("_")
                for i, syllable in enumerate(syllables):
                    sent.append(syllable)
                    if i == 0:
                        sent_labels.append('B_W')
                    else:
                        sent_labels.append('I_W')
            sentences.append(sent)
            labels.append(sent_labels)

    return sentences, labels


def load_data_from_dir(data_path):
    """
    Load data from files in a directory. This function uses load_data_from_file
    :param data_path: path to directory
    :return: combined sentences array and label array
    """
    file_names = os.listdir(data_path)
    sentences = None
    labels = None
    for f_name in file_names:
        file_path = os.path.join(data_path, f_name)
        if f_name.startswith('.') or os.path.isdir(file_path):
            continue
        batch_sentences, batch_labels = load_data_from_file(file_path)
        if sentences is None:
            sentences = batch_sentences
            labels = batch_labels
        else:
            sentences += batch_sentences
            labels += batch_labels
    return sentences, labels


class CrfTokenizer(BaseTokenizer):
    def __init__(self, config_root_path="", bi_grams_path='bi_grams.txt', tri_grams_path='tri_grams.txt',
                 crf_config_path='crf_config.txt',
                 features_path='crf_features.txt',
                 model_path='vi-segmentation.crfsuite',
                 load_data_f_file=load_data_from_dir,
                 base_lib='sklearn_crfsuite'):
        """
        Initial config
        :param config_root_path: path to directory where you put config files such as bi_grams.txt, tri_grams.txt, ...
        :param bi_grams_path: path to bi-grams set
        :param tri_grams_path: path to tri-grams set
        :param crf_config_path: path to crf model config file
        :param features_path: path to feature config file
        :param model_path: path to save or load model to/from file
        :param load_data_f_file: method using to load data from file to return sentences and labels
        :param base_lib: library to use for CRF algorithm, default: sklearn_crfsuite, other choices are pycrfsuite
        """
        self.bi_grams = load_n_grams(config_root_path + bi_grams_path)
        self.tri_grams = load_n_grams(config_root_path + tri_grams_path)
        self.crf_config = load_crf_config(config_root_path + crf_config_path)
        self.features_cfg_arr = load_crf_config(config_root_path + features_path)
        self.center_id = int((len(self.features_cfg_arr) - 1) / 2)
        self.function_dict = {
            'bias': lambda word, *args: 1.0,
            'word.lower()': lambda word, *args: word.lower(),
            'word.isupper()': lambda word, *args: word.isupper(),
            'word.istitle()': lambda word, *args: word.istitle(),
            'word.isdigit()': lambda word, *args: word.isdigit(),
            'word.bi_gram()': lambda word, word1, relative_id, *args: self._check_bi_gram([word, word1], relative_id),
            'word.tri_gram()': lambda word, word1, word2, relative_id, *args: self._check_tri_gram(
                [word, word1, word2], relative_id)
        }
        self.model_path = model_path
        self.load_data_from_file = load_data_f_file
        self.tagger = None
        self.base_lib = base_lib

    def _check_bi_gram(self, a, relative_id):
        """
        Check if bi-gram exists in dictionary
        :param a: list of words [word1, word]. Note: word1 can be next word or previous word but same format
        :param relative_id: relative id compare to current word, e.g. -1 = previous word, +1 = next word
        :return: True or False
        """
        if relative_id < 0:
            return ' '.join([a[0], a[1]]).lower() in self.bi_grams
        else:
            return ' '.join([a[1], a[0]]).lower() in self.bi_grams

    def _check_tri_gram(self, b, relative_id):
        """
        Check if tri-gram exists in dictionary
        :param b: list of words [word2, word1, word]. Note: word2 can be next next word or
         previous previous word but same format
        :param relative_id: relative id compare to current word, e.g. -2 = previous previous word, +2 = next next word
        :return: True or False
        """
        if relative_id < 0:
            return ' '.join([b[0], b[1], b[2]]).lower() in self.tri_grams
        else:
            return ' '.join([b[2], b[1], b[0]]).lower() in self.tri_grams

    def _get_base_features(self, features_cfg_arr, word_list, relative_id=0):
        """
        Calculate each feature one by one
        :param features_cfg_arr: array of feature names
        :param word_list: related word list, word word+1, ...
        :param relative_id: relative id compare to target word, E.g. -2, -1, 0, +1, +2
        :return: dictionary of features
        """
        prefix = ""
        if relative_id < 0:
            prefix = str(relative_id) + ":"
        elif relative_id > 0:
            prefix = '+' + str(relative_id) + ":"

        features_dict = dict()
        for ft_cfg in features_cfg_arr:
            features_dict.update({prefix+ft_cfg: wrapper(self.function_dict[ft_cfg], word_list + [relative_id])})
        return features_dict

    @staticmethod
    def _check_special_case(word_list):
        """
        Check if we catch a special case
        :param word_list: [previous_word, current_word]
        :return: True or False
        """
        # Current word start with upper case but previous word NOT
        if word_list[1].istitle() and (not word_list[0].istitle()):
            return True

        # Is a punctuation
        for word in word_list:
            if word in string.punctuation:
                return True
        # Is a digit
        for word in word_list:
            if word[0].isdigit():
                return True

        return False

    def create_syllable_features(self, text, word_id):
        """
        Calculate features of a word in a text data (matrix format or vector format)
        :param text: input text in form of array-like
        :param word_id: position of word in text
        :return: dictionary of full features
        """
        word = text[word_id]
        features_dict = self._get_base_features(self.features_cfg_arr[self.center_id], [word])

        if word_id > 0:
            word1 = text[word_id - 1]
            features_dict.update(self._get_base_features(self.features_cfg_arr[self.center_id - 1],
                                                         [word1, word], -1))
            if word_id > 1:
                word2 = text[word_id - 2]
                features_dict.update(self._get_base_features(self.features_cfg_arr[self.center_id - 2],
                                                             [word2, word1, word], -2))
        if word_id < len(text) - 1:
            word1 = text[word_id + 1]
            features_dict.update(self._get_base_features(self.features_cfg_arr[self.center_id + 1],
                                                         [word1, word], +1))
            if word_id < len(text) - 2:
                word2 = text[word_id + 2]
                features_dict.update(self._get_base_features(self.features_cfg_arr[self.center_id + 2],
                                                             [word2, word1, word], +2))
        return features_dict

    def create_sentence_features(self, prepared_sentence):
        """
        Create features for a sentence
        :param prepared_sentence: input sentence. E.g. ['Hello', 'World'] is a sentence = list of words
        :return: List of features. E.g. [{'bias': 1.0, 'lower': 'hello'}, {'bias': 1.0, 'lower': 'world'}]
        """
        return [self.create_syllable_features(prepared_sentence, i) for i in range(len(prepared_sentence))]

    def prepare_training_data(self, sentences, labels):
        """
        Prepare the correct format from sentences's features and labels
        :param sentences: input sentences's features
        :param labels: labels of words
        :return: Array of sentence, sentence is now a list of feature dictionarys,
        each dictionary is corresponding to a word
        """
        X = []
        y = []
        for i, sent in enumerate(sentences):
            X.append(self.create_sentence_features(sent))
            y.append(labels[i])
        return X, y

    def train(self, data_path):
        """
        Train train data loaded from file and save model to model_path
        :param data_path: path to data file or directory depending on self.load_data_from_file method
        :return: None
        """
        sentences, labels = self.load_data_from_file(data_path)
        X, y = self.prepare_training_data(sentences, labels)

        if self.base_lib == "sklearn_crfsuite":
            crf = sklearn_crfsuite.CRF(
                algorithm=self.crf_config['algorithm'],
                c1=self.crf_config['c1'],
                c2=self.crf_config['c2'],
                max_iterations=self.crf_config['max_iterations'],
                all_possible_transitions=self.crf_config['all_possible_transitions']
            )
            crf.fit(X, y)
            # joblib.dump(crf, self.model_path)
            with open(self.model_path, 'wb') as fw:
                pickle.dump(crf, fw)
        else:
            trainer = pycrfsuite.Trainer(verbose=False)

            for xseq, yseq in zip(X, y):
                trainer.append(xseq, yseq)

            trainer.set_params(self.crf_config)
            trainer.train(self.model_path)

    def load_tagger(self):
        """
        Load tagger model from file
        :return: None
        """
        print("Loading model from file {}".format(self.model_path))
        if self.base_lib == "sklearn_crfsuite":
            # self.tagger = joblib.load(self.model_path)
            with open(self.model_path, 'rb') as fr:
                self.tagger = pickle.load(fr)
        else:
            self.tagger = pycrfsuite.Tagger()
            self.tagger.open(self.model_path)

    def tokenize(self, text):
        """
        Tokenize sentence using trained crf model
        :param text: input text sentence
        :return: list of words
        """
        if self.tagger is None:
            self.load_tagger()
        sent = self.syllablize(text)
        syl_len = len(sent)
        if syl_len <= 1:
            return sent
        test_features = self.create_sentence_features(sent)
        if self.base_lib == "sklearn_crfsuite":
            prediction = self.tagger.predict([test_features])[0]
        else:
            prediction = self.tagger.tag(test_features)
        word_list = []
        pre_word = sent[0]
        for i, p in enumerate(prediction[1:], start=1):
            if p == 'I_W' and not self._check_special_case(sent[i-1:i+1]):
                pre_word += "_" + sent[i]
                if i == (syl_len - 1):
                    word_list.append(pre_word)
            else:
                if i > 0:
                    word_list.append(pre_word)
                if i == (syl_len - 1):
                    word_list.append(sent[i])
                pre_word = sent[i]

        return word_list

    def get_tokenized(self, text):
        """
        Convert text to tokenized text using trained crf model
        :param text: input text
        :return: tokenized text
        """
        if self.tagger is None:
            self.load_tagger()
        sent = self.syllablize(text)
        if len(sent) <= 1:
            return text
        test_features = self.create_sentence_features(sent)
        if self.base_lib == "sklearn_crfsuite":
            prediction = self.tagger.predict([test_features])[0]
        else:
            prediction = self.tagger.tag(test_features)
        complete = sent[0]
        for i, p in enumerate(prediction[1:], start=1):
            if p == 'I_W' and not self._check_special_case(sent[i-1:i+1]):
                complete = complete + '_' + sent[i]
            else:
                complete = complete + ' ' + sent[i]
        return complete


"""Tests"""


def test_base():
    params = load_crf_config('crf_config.txt')
    print(params)
    sents = [["Thuế", "thu", "nhập", "cá", "nhân"]]
    crf_tokenizer_obj = CrfTokenizer(load_data_f_file=load_data_from_file)

    fdict = crf_tokenizer_obj.create_syllable_features(sents[0], word_id=2)
    print(fdict)
    test_sent = "Thuế thu nhập cá nhân"
    crf_tokenizer_obj.train('data.txt')
    tokenized_sent = crf_tokenizer_obj.get_tokenized(test_sent)
    print(tokenized_sent)
    tokens = crf_tokenizer_obj.tokenize(test_sent)
    print(tokens)


def test():
    crf_tokenizer_obj = CrfTokenizer()
    test_sent = "Thuế thu nhập cá nhân"
    crf_tokenizer_obj.train('../data/tokenized/samples/training')
    tokenized_sent = crf_tokenizer_obj.get_tokenized(test_sent)
    print(tokenized_sent)
    tokens = crf_tokenizer_obj.tokenize(test_sent)
    print(tokens)

if __name__ == '__main__':
    test()
