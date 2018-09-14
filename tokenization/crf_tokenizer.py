import ast
from base_tokenizer import BaseTokenizer
from utils import load_n_grams
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


class CrfTokenizer(BaseTokenizer):
    def __init__(self, bi_grams_path='bi_grams.txt', tri_grams_path='tri_grams.txt', features_path='crf_features.txt'):
        """
        Initial config
        :param bi_grams_path: path to bi-grams set
        :param tri_grams_path: path to tri-grams set
        """
        self.bi_grams = load_n_grams(bi_grams_path)
        self.tri_grams = load_n_grams(tri_grams_path)
        self.features_cfg_arr = load_crf_config(features_path)
        self.center_id = int((len(self.features_cfg_arr) - 1) / 2)
        self.function_dict = {
            'bias': lambda word, *args: 1.0,
            'lower': lambda word, *args: word.lower(),
            'isupper': lambda word, *args: word.isupper(),
            'istitle': lambda word, *args: word.istitle(),
            'isdigit': lambda word, *args: word.isdigit(),
            'bi_gram': lambda word, word1, relative_id, *args: self._check_bi_gram([word, word1], relative_id),
            'tri_gram': lambda word, word1, word2, relative_id, *args: self._check_tri_gram(
                [word, word1, word2], relative_id)
        }

    def _check_bi_gram(self, a, relative_id):
        if relative_id < 0:
            return ' '.join([a[0], a[1]]).lower() in self.bi_grams
        else:
            return ' '.join([a[1], a[0]]).lower() in self.bi_grams

    def _check_tri_gram(self, b, relative_id):
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

    def create_char_features(self, text, word_id):
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


def test():
    from pyvi import ViTokenizer
    ViTokenizer.tokenize("Thuế thu nhập cá nhân")
    params = load_crf_config('crf_config.txt')
    print(params)
    fdict = CrfTokenizer().create_char_features(["Thuế", "thu", "nhập", "cá", "nhân"], word_id=2)
    print(fdict)

if __name__ == '__main__':
    test()
