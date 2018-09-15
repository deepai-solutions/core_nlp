
# Core NLP algorithms for Vietnamese

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
test_sent = "Thuế thu nhập cá nhân"
# Note: If you trained your model, please set correct model path and do not train again!
crf_tokenizer_obj.train('../data/tokenized/samples/training')
tokenized_sent = crf_tokenizer_obj.get_tokenized(test_sent)
print(tokenized_sent)
```
Output will be an array: ['Thuế_thu_nhập', 'cá_nhân']

