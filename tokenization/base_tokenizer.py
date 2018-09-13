import sys
import re
import unicodedata as ud
__author__ = "Ha Cao Thanh"
__copyright__ = "Copyright 2018, DeepAI-Solutions"


class BaseTokenizer(object):
    def tokenize(self, text):
        """
        Convert a sentence to an array of words
        :param text: input sentence (format: string)
        :return: array of words (format: array of strings)
        """
        pass

    def get_tokenized(self, text):
        pass

    @staticmethod
    def syllablize(text):
        """
        Split a sentences into an array of syllables
        :param text: input sentence
        :return:
        """
        text = ud.normalize('NFC', text)

        specials = ["==>", "->", "\.\.\.", ">>"]
        digit = "\d+([\.,_]\d+)+"
        email = "(^[a-zA-Z0-9_.+-]+@[a-zA-Z0-9-]+\.[a-zA-Z0-9-.]+$)"
        web = "^(http[s]?://)?(?:[a-zA-Z]|[0-9]|[$-_@.&+]|[!*\(\),]|(?:%[0-9a-fA-F][0-9a-fA-F]))+$"
        datetime = [
            "\d{1,2}\/\d{1,2}(\/\d+)?",
            "\d{1,2}-\d{1,2}(-\d+)?",
        ]
        word = "\w+"
        non_word = "[^\w\s]"
        abbreviations = [
            "[A-ZĐ]+\.",
            "Tp\.",
            "Mr\.", "Mrs\.", "Ms\.",
            "Dr\.", "ThS\."
        ]

        patterns = []
        patterns.extend(abbreviations)
        patterns.extend(specials)
        patterns.extend([web, email])
        patterns.extend(datetime)
        patterns.extend([digit, non_word, word])

        patterns = "(" + "|".join(patterns) + ")"
        if sys.version_info < (3, 0):
            patterns = patterns.decode('utf-8')
        tokens = re.findall(patterns, text, re.UNICODE)

        return [token[0] for token in tokens]

