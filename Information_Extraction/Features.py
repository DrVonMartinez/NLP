import random

learned_words: dict[str, list[str]] = {}


def current_word(word: str) -> bool:
    return word


def previous_word_location(words: list[str]):
    return words[0] in ['at', 'to']


def previous_word_date(words: list[str]):
    return words[0] in ['at']


def previous_word(words: list[str]):
    return words[0]


def next_word(words: list[str]):
    return words[-1]


def label_context_prior(words: list[str], distance: int = 1):
    if len(words) < distance + 1:
        raise ValueError('Context is too small')


def has_digit(word: str):
    return any(map(lambda x: str(x).isdigit(), word))


class Feature:
    def __init__(self, class_name: str, func):
        self.__class_name: str = class_name
        self.___datum_ = func
        self.__weight: float = random.random()
        self.__name = str(func.__name__)

    def feature(self, class_name: str, args) -> bool:
        return self.___datum_(args) and class_name == self.__class_name

    @property
    def class_name(self) -> str:
        return self.__class_name

    @property
    def weight(self) -> float:
        return self.__weight

    @weight.setter
    def weight(self, weight: float):
        self.__weight = weight

    def __str__(self):
        return self.__name + ', ' + self.__class_name

    def __eq__(self, other):
        if not isinstance(other, Feature):
            return False
        name_check = str(other) == self.__name
        class_check = other.class_name == self.__class_name
        func_check = other.___datum_ == self.___datum_
        return all([name_check, class_check, func_check])
