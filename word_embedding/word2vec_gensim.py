from gensim.test.utils import common_texts
from gensim.models import Word2Vec
import os


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
    sentences = load_data(data_path)
    model = Word2Vec(sentences, size=100, window=5, min_count=1, workers=4)
    model.save(model_path)


def test(model_path="../models/word2vec.model"):
    model = Word2Vec.load(model_path)
    vector = model.wv['thu_nhập']
    print(vector)


if __name__ == '__main__':
    # train()
    test()



