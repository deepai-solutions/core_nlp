import numpy as np
import keras
from keras.models import Sequential
from keras.layers import Dense, Dropout, LSTM, Bidirectional


class BaseTextClassifier(object):
    def train(self, X, y):
        """
        Training using data X, y
        :param X: input feature array
        :param y: label array
        :return:
        """
        pass

    def predict(self, X):
        """
        Predict for input X
        :param X: input feature array
        :return: y as a label array
        """
        pass


class KerasTextClassifier(BaseTextClassifier):
    def __init__(self, tokenizer, word2vec, model_path, max_length=20, n_epochs=15, batch_size=6, n_class=2):
        self.tokenizer = tokenizer
        self.word2vec = word2vec
        self.word_dim = self.word2vec[self.word2vec.index2word[0]].shape[0]
        self.max_length = max_length
        self.n_epochs = n_epochs
        self.batch_size = batch_size
        self.model_path = model_path
        self.n_class = n_class
        self.model = None

    def train(self, X, y):
        self.model = self.build_model(input_dim=(X.shape[1], X.shape[2]))
        self.model.fit(X, y, batch_size=self.batch_size, epochs=self.n_epochs)
        self.model.save_weights(self.model_path)

    def predict(self, X):
        if self.model is None:
            self.load_model()
        y = self.model.predict(X)
        return y

    def classify(self, sentences):
        X = [sent.strip() for sent in sentences]
        X, _ = self.tokenize_sentences(X)
        X = self.word_embed_sentences(X, max_length=self.max_length)
        y = self.predict(np.array(X))
        print(y)
        y = np.argmax(y, axis=1)
        print(y)
        labels = []
        for lab_ in y:
            if lab_ == 0:
                labels.append('positive')
            else:
                labels.append('negative')
        return labels

    def load_model(self):
        self.model = self.build_model((self.max_length, self.word_dim))
        self.model.load_weights(self.model_path)

    def build_model(self, input_dim):
        model = Sequential()

        model.add(LSTM(64, return_sequences=True, input_shape=input_dim))
        model.add(Dropout(0.2))
        model.add(LSTM(32))
        model.add(Dropout(0.2))
        model.add(Dense(self.n_class, activation="softmax"))

        model.compile(loss=keras.losses.categorical_crossentropy,
                      optimizer=keras.optimizers.Adadelta(),
                      metrics=['accuracy'])
        return model

    def load_data(self, positive_path, negative_path):
        positive_sentences = self.load_data_from_file(positive_path)
        negative_sentences = self.load_data_from_file(negative_path)
        X = positive_sentences + negative_sentences
        y = [[1.0, 0.0] for _ in range(0, len(positive_sentences))] + \
            [[0.0, 1.0] for _ in range(0, len(negative_sentences))]

        X, max_length = self.tokenize_sentences(X)
        X = self.word_embed_sentences(X, max_length=self.max_length)
        return np.array(X), np.array(y)

    def word_embed_sentences(self, sentences, max_length=20):
        embed_sentences = []
        for sent in sentences:
            embed_sent = []
            for word in sent:
                if word.lower() in self.word2vec:
                    embed_sent.append(self.word2vec[word.lower()])
                else:
                    embed_sent.append(np.zeros(shape=(self.word_dim,), dtype=float))
            if len(embed_sent) > max_length:
                embed_sent = embed_sent[:max_length]
            elif len(embed_sent) < max_length:
                embed_sent = np.concatenate((embed_sent, np.zeros(shape=(max_length - len(embed_sent),
                                                                         self.word_dim), dtype=float)),
                                            axis=0)
            embed_sentences.append(embed_sent)
        return embed_sentences

    def tokenize_sentences(self, sentences):
        tokens_list = []
        max_length = -1
        for sent in sentences:
            tokens = self.tokenizer.tokenize(sent)
            tokens_list.append(tokens)
            if len(tokens) > max_length:
                max_length = len(tokens)

        return tokens_list, max_length

    @staticmethod
    def load_data_from_file(file_path):
        with open(file_path, 'r') as fr:
            sentences = fr.readlines()
            sentences = [sent.strip() for sent in sentences if len(sent.strip()) > 0]
        return sentences


class BiDirectionalLSTMClassifier(KerasTextClassifier):
    def build_model(self, input_dim):
        model = Sequential()

        model.add(Bidirectional(LSTM(64, return_sequences=True), input_shape=input_dim))
        model.add(Dropout(0.2))
        model.add(Bidirectional(LSTM(32)))
        model.add(Dropout(0.2))
        model.add(Dense(self.n_class, activation="softmax"))

        model.compile(loss=keras.losses.categorical_crossentropy,
                      optimizer=keras.optimizers.Adadelta(),
                      metrics=['accuracy'])
        return model


def test():
    from tokenization.crf_tokenizer import CrfTokenizer
    from word_embedding.word2vec_gensim import Word2Vec
    word2vec_model = Word2Vec.load('../models/pretrained_word2vec.bin')
    tokenizer = CrfTokenizer(config_root_path='/Users/admin/Desktop/Projects/python/NLP/hactcore/hactcore/nlp/tokenization/',
                             model_path='../models/pretrained_tokenizer.crfsuite')
    # keras_text_classifier = KerasTextClassifier(tokenizer=tokenizer, word2vec=word2vec_model.wv,
    keras_text_classifier = BiDirectionalLSTMClassifier(tokenizer=tokenizer, word2vec=word2vec_model.wv,
                                                        model_path='../models/sentiment_model.h5',
                                                        max_length=20)
    X, y = keras_text_classifier.load_data('../data/sentiment/samples/positive.txt',
                                           '../data/sentiment/samples/negative.txt')

    keras_text_classifier.train(X, y)
    labels = keras_text_classifier.classify(['Dở thế', 'Hay thế', 'Nghe như đấm vào tai'])
    print(labels)

if __name__ == '__main__':
    test()

