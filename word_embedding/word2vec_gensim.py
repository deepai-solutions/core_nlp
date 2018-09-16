from gensim.models import Word2Vec
import os
__author__ = "Cao Bot"
__copyright__ = "Copyright 2018, DeepAI-Solutions"


def load_data_from_file(data_path):
    """
    Load data from file
    :param data_path: input file path
    :return: sentences and labels
    Examples
    sentences = [['Hello', 'World'], ['Hello', 'World']]
    """
    sentences = []
    with open(data_path, 'r') as fr:
        lines = fr.readlines()
        for line in lines:
            sent = line.strip().split()
            sentences.append(sent)

    return sentences


def load_data_from_dir(data_path):
    """
    Load data from directory which contains multiple files
    :param data_path: path to data directory
    :return: 2D-array of sentences
    """
    file_names = os.listdir(data_path)
    sentences = None
    for f_name in file_names:
        file_path = os.path.join(data_path, f_name)
        if f_name.startswith('.') or os.path.isdir(file_path):
            continue
        batch_sentences = load_data_from_file(file_path)
        if sentences is None:
            sentences = batch_sentences
        else:
            sentences += batch_sentences
    return sentences


def train(data_path="../data/word_embedding/samples/training", load_data=load_data_from_dir,
          model_path="../models/word2vec.model"):
    """
    Train data loaded from a file or a directory
    :param data_path: path to a file or to a directory which contains multiple files
    :param load_data: function to load data (from a file or a directory)
    :param model_path: path to save model as a file
    :return: None
    """
    sentences = load_data(data_path)
    model = Word2Vec(sentences, size=100, window=5, min_count=1, workers=4)
    model.save(model_path)


def test(model_path="../models/word2vec.model", word="thu_nhập"):
    """
    Test word2vec model
    :param model_path: path to model file
    :param word: word to test
    :return: None
    """
    model = Word2Vec.load(model_path)
    vector = model.wv[word]
    print(vector)
    sim_words = model.wv.most_similar(word)
    print(sim_words)


if __name__ == '__main__':
    # train()
    test(model_path="../models/pretrained_word2vec.bin")



