# Simple Task
#   Is attitude of the text positive or negative?
# More Complex:
#   Rank attitude of text from 1 to 5
# Advanced:
# Detect the target, source, or complex attitude types
import glob

import nltk
from nltk.corpus import brown
import numpy as np

from Classification import pathing


def is_word(word):
    char_list = [".", "-", "'"]
    while char_list:
        word = word.replace(char_list.pop(0), '')
    return str(word).isalnum()


def training_set(size: int) -> list[tuple[str]]:
    words = list(filter(is_word, map(str.lower, brown.words())))
    return [tuple(words[i: i + size]) for i in range(len(words) - size)]


def tokenization():
    """
    Issues:
        Deal with HTML and XML markup
        Twitter mark-up (names, hash tags)
        Capitalization (preserve for words in all caps)
        Phone Numbers, dates
        Emoticons

    :return:
    """
    pass


def next_punctuation(sentence: list[str]) -> int:
    check = list(map(lambda y: '.' in y or ',' in y or '?' in y or '!' in y, sentence))
    indices = [len(sentence)]
    if any(check):
        indices = list(filter(lambda x: x is not None, [i if check[i] is True else None for i in range(len(check))]))
    return min(indices)


def negation(sentence: str):
    split_sentence = sentence.split(' ')
    check = list(map(lambda x: "n't" in x or 'not' in x, split_sentence))
    indices = []
    if any(check):
        indices = list(filter(lambda x: x is not None, [i if check[i] is True else None for i in range(len(check))]))
    for index in indices:
        try:
            outdex = next_punctuation(split_sentence[index:])
            new_sentence = ['NOT_' + i for i in split_sentence[index + 1: index + outdex]]
        except ValueError:
            continue
        split_sentence[index + 1:index + outdex] = new_sentence
    changed = True
    while changed:
        changed = False
        for i in range(len(split_sentence)):
            if "NOT_NOT_" in split_sentence[i]:
                split_sentence[i] = split_sentence[i][8:]
                changed = True
    return ' '.join(split_sentence)


if __name__ == '__main__':
    negation_test = "didn't not like this movie , but I"
    # print(next_punctuation(negation_test.split(' ')))


def cross_validation_binary(data: list[list[str]], num_folds: int):
    folds = []
    last_fold = 0
    fold_size = len(data[0]) // num_folds
    for i in range(num_folds):
        class_1 = data[0][last_fold: last_fold + fold_size]
        class_2 = data[1][last_fold: last_fold + fold_size]
        folds.append((class_1, class_2))
        last_fold = 1 // num_folds
    return folds


if __name__ == '__main__':
    positive = list(glob.glob(pathing.paths + 'pos\\*.txt'))
    negative = list(glob.glob(pathing.paths + 'neg\\*.txt'))
    training_sets = cross_validation_binary([positive, negative], 10)


def pointwise_mutual_information(word1: str, word2: str, training: list[tuple[str]]) -> float:
    def hits(word):
        return len(list(filter(lambda u: word in u, training)))

    def near(word_u, word_v):
        u = word_u == word1 and word_v == word2
        v = word_u == word2 and word_v == word1
        return u or v

    hits_word1 = hits(word1)
    hits_word2 = hits(word2)
    near_hits = len(list(filter(lambda a: near(a[0], a[1]), training)))
    print(near_hits, hits_word1, hits_word2)
    return np.log2(near_hits) - (np.log2(hits_word1) + np.log2(hits_word2))


def polarity(phrase: tuple[str], training: list[str]):
    def hits(word):
        return max(len(list(filter(lambda u: word in u, training))), 0.5)

    def near(term):
        return max(int(term in phrase), 1)

    return np.log2((near("excellent") * hits('poor')) / (near("poor") * hits('excellent')))


def is_turney_pos(word1: str, word2: str, word3: str) -> bool:
    tags = {x: y for x, y in nltk.pos_tag([word1, word2, word3])}
    # print(tags)

    if tags[word1] == 'JJ':
        if tags[word2] == 'JJ':
            return tags[word3] not in ['NN', 'NNS']
        return tags[word2] in ['NN', 'NNS']
    elif tags[word1] in ['RB', 'RBR', 'RBS']:
        if tags[word2] == 'JJ':
            return tags[word3] not in ['NN', 'NNS']
        return tags[word2] in ['VB', 'VBD', 'VBN', 'VBG']
    return tags[word1] not in ['NN', 'NNS'] and tags[word2] == 'JJ' and tags[word3] not in ['NN', 'NNS']


if __name__ == '__main__':
    print(is_turney_pos('Ran', 'quickly', 'here'))
    # assert False
    file_sets = training_sets[0][0] + training_sets[0][1]
    files = []

    for file in file_sets:
        with open(file, 'r') as reader:
            lines = (' '.join(reader.readlines())).split(' ')
            words = list(filter(is_word, map(str.lower, lines)))
            # print("Words", words)
            triples = [tuple(words[i: i + 3]) for i in range(len(words) - 3)]
            phrases = list(filter(lambda x: is_turney_pos(*x), triples))
            mapped_phrases = list(map(lambda x: is_turney_pos(*x), triples))
            # print(mapped_phrases)
            # print(phrases)
            polarity_values = [polarity(_phrase_, words) for _phrase_ in phrases]
            if np.average(polarity_values) > 0:
                print("Polarity:", np.average(polarity_values), mapped_phrases)
