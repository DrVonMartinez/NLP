from collections import Counter
import networkx as nx
# from N_grams import higher_order_kneser_ney, good_turing, unigram_prior, n_gram
# from N_grams import higher_order_kneser_ney
from Spelling_Correction import one_step_spelling
from nltk.corpus import brown
import numpy as np


def is_word(word):
    char_list = [".", "-", "'"]
    while char_list:
        word = word.replace(char_list.pop(0), '')
    return str(word).isalnum()


word_set = Counter(filter(is_word, map(str.lower, brown.words())))
training_set = ' '.join(list(filter(is_word, map(str.lower, brown.words()))))
higher_order_sentence = [' '.join(brown.words()[j: j + 2]) for j in range(len(brown.words()) - 2)]

print(higher_order_sentence)
assert False


def simple_training(n) -> tuple[dict, dict]:
    training = list(filter(is_word, map(str.lower, brown.words())))
    gram: dict[str] = {}
    word_count: dict[str] = {}
    for i in range((n - 1), len(training)):
        gram.setdefault(' '.join(training[i - (n - 1): i + 1]), 0)
        gram[' '.join(training[i - (n - 1): i + 1])] += 1
    for i in range(len(training)):
        word_count.setdefault(training[i], 0)
        word_count[training[i]] += 1
    return word_count, gram


def _unigram_(sentence: str):
    word_count = Counter(word_set)

    def n_phrase(phrase):
        try:
            return np.log((word_count[phrase] + 1) / (word_count[phrase.split(' ')[-1]] + len(word_count.values())))
        except KeyError:
            return np.log(1 / len(word_count.values()))

    return np.exp(sum(map(n_phrase, sentence.split(' '))))


def higher_order_kneser_ney(sentence, discount=0.75) -> float:
    word_count, gram = simple_training(2)

    def continuation(word):
        continuation_dict = {}
        num_word_count, num_word_gram = simple_training(len(word))
        for i in num_word_gram:
            split: list[str] = i.split(' ')
            words, word_i = split[:-1], split[-1]
            if word_i == word[-1]:
                continuation_dict.setdefault(' '.join(words), 0)
                continuation_dict[' '.join(words)] += 1
        return len(continuation_dict) / len(num_word_gram)

    def interpolation_weight(word):
        num_word_count, num_word_gram = simple_training(len(word))
        interpolation_dict = {}
        for i in num_word_gram:
            split = i.split(' ')
            words, word_i = split[:-1], split[-1]
            if word_i == word[-1]:
                interpolation_dict.setdefault(word_i, 0)
                interpolation_dict[word_i] += 1
        print(interpolation_dict)
        return discount * len(interpolation_dict) / num_word_count[word[-1]]

    def n_phrase(phrase, num):
        words = phrase.split(' ')
        num_word_count, _ = simple_training(num)
        theta = interpolation_weight(words[-1 * num:]) * continuation(words[-1 * (num + 1):])
        print(interpolation_weight(words[-1 * num:]), words[-1 * num:])
        try:
            x = (gram[phrase] - discount) / sum(num_word_count.values())
        except KeyError:
            x = 0
        print(max(x, 0) + theta)
        return np.log(max(x, 0) + theta)
    result = n_set(sentence)
    return np.exp(sum(map(n_phrase, result, [2] * len(result))))


def noisy_channel_graph(noisy_sentence: str, misspell_rate=0.95):
    words = noisy_sentence.split(' ')
    graph = nx.Graph(name='Noisy').to_directed()
    prior_words = []
    for word in words:
        alternate_spellings = one_step_spelling(word, misspell_rate=misspell_rate)
        print(alternate_spellings)
        for alternate_spelling in alternate_spellings:
            graph.add_node(alternate_spelling, weight=alternate_spellings[alternate_spelling])
            for prior_word in prior_words:
                graph.add_edge(prior_word, alternate_spelling)
        prior_words = alternate_spellings
    print(nx.adjacency_matrix(graph))
    raise NotImplementedError('Not complete yet')


def noisy_channel_pairs(noisy_sentence: str, misspell_rate=0.90):
    words = noisy_sentence.split(' ')
    denoised_words = ''
    for word in words:
        alternate_spellings = one_step_spelling(word, misspell_rate=misspell_rate)
        alternate_spelling_seq = []
        distinct_words = []
        alternate_words = []
        conclusions = []
        for alternate_spelling in alternate_spellings:
            # unigram = _unigram_(alternate_spelling)
            n_gram = _unigram_(alternate_spelling)
            if alternate_spellings[alternate_spelling] > 0 and n_gram > 0:
                alternate_spelling_seq.append(n_gram * alternate_spellings[alternate_spelling])
                distinct_words.append((alternate_spelling, n_gram))
                alternate_words.append((alternate_spelling, alternate_spellings[alternate_spelling]))
                conclusions.append((alternate_spelling, alternate_spelling_seq[-1]))
        index = np.argmax(np.array(alternate_spelling_seq) * 10**9)
        # print(sorted(distinct_words, key=lambda x: x[0]), distinct_words[index])
        # print(sorted(alternate_words, key=lambda x: x[0]), distinct_words[index])
        # print(sorted(conclusions, key=lambda x: x[0]), distinct_words[index])
        # print(np.array(alternate_spelling_seq) * 10**9, index)
        denoised_words += ' ' + str(distinct_words[index][0])
    return denoised_words


print(noisy_channel_pairs('two of thew'))

