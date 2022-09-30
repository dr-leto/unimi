import typing
from collections import defaultdict
import nltk


class WordIndex(object):
    def __init__(self, corpus:typing.List[str]):
        self.corpus = corpus
        self.index = defaultdict(lambda: 0)

    def read(self):
        for document in self.corpus:
            for word in nltk.word_tokenize(document):
                self.index[word] += 1
