from collections import Counter
import networkx as nx
# from N_grams import higher_order_kneser_ney, good_turing, unigram_prior, n_gram
from N_grams import higher_order_kneser_ney
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


def _unigram_(sentence: str):
    word_count = Counter(word_set)

    def n_phrase(phrase):
        try:
            return np.log((word_count[phrase] + 1) / (word_count[phrase.split(' ')[-1]] + len(word_count.values())))
        except KeyError:
            return np.log(1 / len(word_count.values()))

    return np.exp(sum(map(n_phrase, sentence.split(' '))))


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


def recursive_hokn(words: list[str]):
    if len(words) == 1:
        return higher_order_kneser_ney(' '.join(words), word_set, 1)
    else:
        return recursive_hokn(words[1:]) * higher_order_kneser_ney(' '.join(words), word_set, len(words))


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
            if alternate_spellings[alternate_spelling] > 0 and unigram > 0:
                alternate_spelling_seq.append(unigram * alternate_spellings[alternate_spelling])
                distinct_words.append((alternate_spelling, unigram))
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

