
# Core NLP algorithms for Vietnamese

### Expectation:
We want to contribute knowledge and source codes to develop an active Vietnamese Natural Language Processing community.
Beginners don't have to build basic code again. 
They can use code for implementing in their system or doing research for improving performance. 

## Modules
## 1. Tokenizer
Function: Convert text into tokens
E.g.

```python
from tokenization.dict_models import LongMatchingTokenizer
lm_tokenizer = LongMatchingTokenizer()
tokens = lm_tokenizer.tokenize("Thuế thu nhập cá nhân")
```
Output will be an array: ['Thuế_thu_nhập', 'cá_nhân']

Example 2: Using crf

```python
from tokenization.crf_tokenizer import CrfTokenizer

crf_tokenizer_obj = CrfTokenizer()
crf_tokenizer_obj.train('data/tokenized/samples/training')
# Note: If you trained your model, please set correct model path and do not train again!
# crf_tokenizer_obj = CrfTokenizer(model_path='models/pretrained_tokenizer.crfsuite')

test_sent = "Thuế thu nhập cá nhân"
tokenized_sent = crf_tokenizer_obj.get_tokenized(test_sent)
print(tokenized_sent)
```
Output will be an array: ['Thuế_thu_nhập', 'cá_nhân']

## 2. Word Embedding

Download data from websites such as Wikipedia, Vietnamese News. You can use any method and here we propose one for you:
```python
from word_embedding.utils import download_html
url_path = "https://dantri.com.vn/su-kien/anh-huong-bao....htm"
output_path = "data/word_embedding/real/html/html_data.txt"
download_html(url_path, output_path, should_clean=True)
```

Clean and tokenize text in files. We propose a method to clean each file in a directory.
```python
input_dir = 'data/word_embedding/real/html'
output_dir = 'data/word_embedding/real/training'
from tokenization.crf_tokenizer import CrfTokenizer
from word_embedding.utils import clean_files_from_dir
crf_config_root_path = "tokenization/"
crf_model_path = "models/pretrained_tokenizer.crfsuite"
tokenizer = CrfTokenizer(config_root_path=crf_config_root_path, model_path=crf_model_path)
clean_files_from_dir(input_dir, output_dir, should_tokenize=True, tokenizer=tokenizer)
```

Training
```python
from word_embedding.word2vec_gensim import train
data_path="data/word_embedding/samples/training"
model_path="models/word2vec.model"
train(data_path=data_path, model_path=model_path)
```

Test
```python
from word_embedding.word2vec_gensim import test
model_path = "models/word2vec.model"
test_word = "thu_nhập"
test(model_path=model_path, word=test_word)
```

Use pre-trained word2vec model
```python
model_path = "models/pretrained_word2vec.bin"
```
Other links for pre-trained word2vec models:
https://github.com/sonvx/word2vecVN
https://github.com/Kyubyong/wordvectors


## 3. Text classification
#### (Run: python demo.py)

```python
from tokenization.crf_tokenizer import CrfTokenizer
from word_embedding.word2vec_gensim import Word2Vec
from text_classification.short_text_classifiers import BiDirectionalLSTMClassifier, load_synonym_dict

# Please give the correct paths
word2vec_model = Word2Vec.load('models/pretrained_word2vec.bin')

tokenizer = CrfTokenizer(config_root_path='tokenization/',
                         model_path='models/pretrained_tokenizer.crfsuite')
sym_dict = load_synonym_dict('data/sentiment/synonym.txt')
keras_text_classifier = BiDirectionalLSTMClassifier(tokenizer=tokenizer, word2vec=word2vec_model.wv,
                                                    model_path='models/sentiment_model.h5',
                                                    max_length=10, n_epochs=10,
                                                    sym_dict=sym_dict)
X, y = keras_text_classifier.load_data(['data/sentiment/samples/positive.txt',
                                       'data/sentiment/samples/negative.txt'],
                                       load_method=keras_text_classifier.load_data_from_file)

keras_text_classifier.train(X, y)
label_dict = {0: 'tích cực', 1: 'tiêu cực'}
test_sentences = ['Dở thế', 'Hay thế', 'phim chán thật', 'nhảm quá']
labels = keras_text_classifier.classify(test_sentences, label_dict=label_dict)
print(labels)  # Output: ['tiêu cực', 'tích cực', 'tiêu cực', 'tiêu cực']
```


## Requirements
Please check requirments.txt

For tokenization modules:
- python-crfsuite
- sklearn-crfsuite (default with pre-trained model, but optional. Feel free to use python-crfsuite)

For word_embedding vector:
- tokenization module
- gensim

For text classification
- tokenization module
- word_embedding module
- keras

FYI:
sklearn-crfsuite is a wrapper of python-crfsuite. 
sklearn-crfsuite offers scikit-learn styled API with grid-search for hyper-parameters optimization

