
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
