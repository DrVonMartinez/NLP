from functools import reduce

import nltk
import numpy as np
from nltk.corpus import wordnet
from nltk.corpus import wordnet_ic
import networkx as nx
import abc
import matplotlib.pyplot as plt


class WordSimilarity(abc.ABC):
    def __init__(self):
        self.__word_a: str or None = None
        self.__word_b: str or None = None

    def __str__(self):
        return '({a}, {b})'.format(a=self.__word_a, b=self.__word_b)

    @property
    def _word_a(self) -> str:
        return self.__word_a

    @_word_a.setter
    def _word_a(self, value: str):
        self.__word_a = value

    @property
    def _word_b(self) -> str:
        return self.__word_b

    @_word_b.setter
    def _word_b(self, value: str):
        self.__word_b = value

    @property
    def _synonyms(self) -> dict[str, list]:
        return {self.__word_a: [synonym for synonym in wordnet.synsets(self.__word_a)],
                self.__word_b: [synonym for synonym in wordnet.synsets(self.__word_b)]}


class PathBasedSimilarity(WordSimilarity):
    def __init__(self, word_a: str, word_b: str):
        super().__init__()
        self._word_a = word_a
        self._word_b = word_b
        self.__graph: nx.Graph = nx.Graph()
        self.__build_hypergraph(word_a)
        self.__build_hypergraph(word_b)

    def __build_hypergraph(self, word):
        self.__graph.add_node(word, name=word)
        self.__graph.add_edge(word, word, weight=1)
        for synonym in self._synonyms[word]:
            if synonym.name() not in self.__graph.nodes:
                self.__graph.add_node(synonym.name(), name=synonym.name())
                self.__graph.add_edge(synonym.name(), synonym.name(), weight=1)
            synonym_name = str(synonym.name()).split('.')[0]
            if synonym_name not in self.__graph.nodes:
                self.__graph.add_node(synonym_name, name=synonym_name)
                self.__graph.add_edge(synonym_name, synonym_name, weight=1)
                self.__graph.add_edge(synonym.name(), synonym_name, weight=1)
            self.__graph.add_edge(word, synonym.name(), weight=1)
            self.__find_root(synonym)

    def __find_root(self, word):
        for hypernym in word.hypernyms():
            if hypernym.name() not in self.__graph.nodes:
                self.__graph.add_node(hypernym.name(), name=hypernym)
                self.__graph.add_edge(hypernym.name(), hypernym.name(), weight=1)
            hypernym_name = str(hypernym.name()).split('.')[0]
            if hypernym_name not in self.__graph.nodes:
                self.__graph.add_node(hypernym_name, name=hypernym_name)
                self.__graph.add_edge(hypernym_name, hypernym_name, weight=1)
                self.__graph.add_edge(hypernym.name(), hypernym_name, weight=1)
            self.__graph.add_edge(word.name(), hypernym.name(), weight=1)
            self.__find_root(hypernym)

    def nodes(self):
        return self.__graph.nodes

    def show_graph(self):
        nx.draw(self.__graph, with_labels=True)
        plt.show()

    def path_based_similarity(self) -> float:
        word_sim = 0
        for a in list(map(lambda x: x.name(), self._synonyms[self._word_a])):
            try:
                sim_path = 1 / (len(nx.shortest_path(self.__graph, a, self._word_b, weight='weight')) + 1)
            except nx.exception.NetworkXNoPath:
                sim_path = 0
            word_sim = max(word_sim, sim_path)
        word_sim = max(word_sim, float(self._word_a == self._word_b))
        return word_sim


class LowestCommonSubsumer(WordSimilarity):
    def __init__(self, corpus: str):
        super().__init__()
        self.__corpus = corpus

    def __hyponymn(self, word: str):
        base = word.split('.')[0]
        wordnet_word = list(filter(lambda x: "'" + word + "'" in str(x), wordnet.synsets(base)))[0]
        return wordnet_word.hyponyms()

    def __lowest_common_subsumer(self):
        print(self._synonyms[self._word_a])
        print(self._synonyms[self._word_b])
        hypernym_a: dict[str, list[str]] = {
            t.name(): [t.name()] + [i.name() for i in t.closure(lambda s: s.hypernyms())]
            for t in self._synonyms[self._word_a]}
        hypernym_b: dict[str, list[str]] = {
            t.name(): [t.name()] + [i.name() for i in t.closure(lambda s: s.hypernyms())]
            for t in self._synonyms[self._word_b]}
        min_height = {}
        for a in hypernym_a:
            for b in hypernym_b:
                proposed = set(hypernym_a[a]).intersection(set(hypernym_b[b]))
                if proposed:
                    pathing = sorted([hypernym_a[a].index(i) for i in proposed])
                    min_dex = min(pathing)
                    max_dex = max(pathing)
                    min_height[max_dex - min_dex] = hypernym_a[a][min_dex]
        min_key = max(min_height.keys())
        return min_height[min_key]

    def resnik_similarity(self, word_a, word_b) -> float:
        self._word_a = word_a
        self._word_b = word_b
        l_c_s = self.__lowest_common_subsumer()
        print(l_c_s)
        print(self.__hyponymn(l_c_s))
        return 0

    def nltk_resnik_similarity(self, word_a, word_b, file):
        ic = wordnet_ic.ic(file)
        word__a = wordnet.synsets(word_a)
        word__b = wordnet.synsets(word_b)
        best = -1
        for a in word__a:
            for b in word__b:
                try:
                    results = a.res_similarity(b, ic)
                    if results > best:
                        best = results
                except nltk.corpus.reader.wordnet.WordNetError:
                    continue
        return best


if __name__ == '__main__':
    word_1 = 'bank'
    word_2 = 'slope'
    pbs = PathBasedSimilarity(word_1, word_1)
    pbs2 = PathBasedSimilarity(word_1, word_2)
    print(pbs.path_based_similarity())
    print(pbs2.path_based_similarity())

    with open('Texts\\Anthony And Cleopatra.txt') as reader:
        text1 = ' '.join(reader.readlines())
    with open('Texts\\Coriolanus.txt') as reader:
        text2 = ' '.join(reader.readlines())
    lcs = LowestCommonSubsumer(text1)
    lcs2 = LowestCommonSubsumer(text2)
    # print(lcs.resnik_similarity(word_1, word_1))
    print(lcs.nltk_resnik_similarity(word_1, word_1, 'ic-brown.dat'))
    # print(lcs2.resnik_similarity(word_1, word_2))
    print(lcs2.nltk_resnik_similarity(word_1, word_2, 'ic-brown.dat'))
