import itertools
from functools import reduce

import networkx as nx


class Grammar:
    def __init__(self, ):
        self.__terminal_symbols: set[str] = set()
        self.__non_terminal_symbols: dict[str, dict[str, ]] = {}
        self.__rules: dict[str, dict[str, ]] = {}
        self.__rule_pairs: dict[str, list[tuple[str]]] = {}
        self.__rule_tree_types: dict[str, list[nx.Graph]] = {}

    def __probability_function(self, rule: str):
        if 0 > self.__rules[rule]['prob'] > 1:
            raise ValueError("Rule Probability must be at least 0 and no more than 1")
        return self.__rules[rule]['prob']

    def __validate_probability(self):
        for x in self.__non_terminal_symbols:
            total = 0
            for rule in self.__rules:
                total += self.__probability_function(rule) * self.__rules[rule]['func'](x)
            if total != 1:
                return False
        return True

    def add_terminal_symbol(self, symbol: str):
        self.__terminal_symbols.add(symbol)

    def add_non_terminal_symbol(self, symbol: str, probability: float = 0, word_type: str = ''):
        self.__non_terminal_symbols.setdefault(symbol, {'word_type': []})
        self.__non_terminal_symbols[symbol]['word_type'].append(word_type)
        self.__non_terminal_symbols[symbol]['prob_' + word_type] = probability

    def add_rule(self, rule: None, probability: float = 0, name=None, output: tuple = tuple([])):
        if not name:
            name = rule.__name__
        self.__rules[name] = {'prob': probability, 'func': rule}
        self.__rule_pairs.setdefault(name, [output])
        if rule.__name__ in self.__rules:
            raise KeyError('Rule already exists by that name')

    def add_rules(self, rule: tuple, probability: tuple[float], name: str, output: list[tuple]):
        self.__rules.setdefault(name, {})
        self.__rule_pairs.setdefault(name, [])
        for r in range(len(rule)):
            self.__rules[name].setdefault('prob_' + str(r), probability[r])
            self.__rules[name]['func_' + str(r)] = rule[r]
            self.__rule_pairs[name].append(output[r])

    def add_tree_node(self, node_name: str, probability: tuple[float], children: list[tuple[str]]):
        if len(probability) != len(children):
            raise ValueError('There must be a probability for each set of children')
        self.__rule_tree_types.setdefault(node_name, [])
        for c in range(len(children)):
            child = children[c]
            graph = nx.Graph(name='-'.join(child))
            graph.add_node(node_name, probability=probability[c])
            for ch in range(0, len(child)):
                last = 0
                child_node_base = 'Child_' + child[ch].replace(' ', '_')
                child_node = child_node_base
                while child_node in graph.nodes:
                    last += 1
                    child_node = child_node_base + '_' + str(last)
                graph.add_node(child_node)
                graph.add_edge(node_name, child_node)
            self.__rule_tree_types[node_name].append(graph)

    def show_tree(self, node_name: str):
        for i in range(len(self.__rule_tree_types[node_name])):
            graph = self.__rule_tree_types[node_name][i]
            print(node_name)
            print('\t', graph.nodes)
            print('\t', nx.get_node_attributes(graph, 'probability'))

    def show_rules(self):
        return list(self.__rules.keys())

    def show_rule_tuples(self):
        return [tuple([key, self.__rule_pairs[key]]) for key in self.__rule_pairs]

    def show_non_terminal_words(self):
        return list(self.__non_terminal_symbols)

    def tree_probability(self, s: list[str]):
        queue = ['Sentence']  # Assume it's a sentence
        word_set = [self.__non_terminal_symbols[s[i]]['word_type'] for i in range(len(s))]
        print(word_set)
        combinations = [[q] for q in word_set[0]]
        for word_type in word_set[1:]:
            new_set = []
            for i in range(len(word_type)):
                new_set += [c + [word_type[i]] for c in combinations]
            combinations = new_set
        combinations.sort()
        print(combinations)
        print(len(combinations))
        graph_combinations: list[list[str]] = [list(q.nodes)[1:] for q in self.__rule_tree_types['Sentence']]

        def is_not_simple(a: str):
            return 'Phrase' in a

        for sentence in graph_combinations:
            subtree_nodes = []
            for phrase in sentence:
                print(phrase)
                phrase_name = ' '.join(phrase.split('_')[1:3])
                subtree = self.__rule_tree_types[phrase_name]
                print(subtree)
                subtree_nodes += [[list(map(lambda x: ' '.join(x.split('_')[1:3]), list(q.nodes)[1:]))
                                  for q in self.__rule_tree_types[phrase_name]]]
            total = ['Phrase']
            while all(map(is_not_simple, total)):
                pieces = list(subtree_nodes[0])
                for i in range(1, len(subtree_nodes)):
                    pieces = [list(map(lambda x: x + subtree_nodes[i][j], pieces)) for j in range(len(subtree_nodes[i]))]
                subtree = list(reduce(lambda x, y: x + y, pieces))
                print(subtree)

    def __bool__(self):
        return self.__validate_probability()

    def __contains__(self, item):
        if callable(item):
            return item.__name__ in self.__rules
        else:
            return item in self.__non_terminal_symbols or item in self.__terminal_symbols

    def __add__(self, other):
        if isinstance(other, Grammar):
            self.__non_terminal_symbols += other.__non_terminal_symbols
            self.__terminal_symbols += other.__terminal_symbols
            self.__rules += other.__rules
        else:
            raise ArithmeticError("Trying to combine unlike types " + str(type(other)) + ' and ' + str(type(self)))

    def __radd__(self, other):
        if isinstance(other, Grammar):
            self.__non_terminal_symbols += other.__non_terminal_symbols
            self.__terminal_symbols += other.__terminal_symbols
            self.__rules += other.__rules
        else:
            raise ArithmeticError("Trying to combine unlike types " + str(type(other)) + ' and ' + str(type(self)))


def is_sentence(pos_tags: list[str]):
    return is_noun_phrase(pos_tags) and is_verb_phrase(pos_tags)


def is_verb_phrase(pos_tags: list[str]):
    return is_verb_phrase_a(pos_tags) or is_verb_phrase_b(pos_tags)


def is_verb_phrase_a(pos_tags: list[str]):
    for i in range(len(pos_tags)):
        a = is_verb_phrase(pos_tags[:i])
        b = is_noun_phrase(pos_tags[i:])
        if a and b:
            return True
    return False


def is_verb_phrase_b(pos_tags: list[str]):
    for i in range(len(pos_tags)):
        a = is_verb_phrase(pos_tags[:i])
        for j in range(i + 1, len(pos_tags)):
            c = is_noun_phrase(pos_tags[i:j])
            d = is_prepositional_phrase(pos_tags[j:])
            if a and c and d:
                return True
    return False


def is_prepositional_phrase(pos_tags: list[str]):
    return is_prepositional(pos_tags[0]) and is_noun_phrase(pos_tags[1:])


def is_noun(pos_tag: str):
    return pos_tag == 'N'


def is_verb(pos_tag: str):
    return pos_tag == 'V'


def is_prepositional(pos_tag: str):
    return pos_tag == 'P'


def is_noun_phrase(pos_tags: list[str]):
    return is_noun_phrase_a(pos_tags) or is_noun_phrase_b(pos_tags) or is_noun_phrase_c(pos_tags)


def is_noun_phrase_a(pos_tags: list[str]):
    for i in range(len(pos_tags)):
        a = is_noun_phrase(pos_tags[:i])
        b = is_noun_phrase(pos_tags[i:])
        if a and b:
            return True
    return False


def is_noun_phrase_b(pos_tags: list[str]):
    for i in range(len(pos_tags)):
        a = is_noun_phrase(pos_tags[:i])
        c = is_prepositional_phrase(pos_tags[i:])
        if a and c:
            return True
    return False


def is_noun_phrase_c(pos_tags: list[str]):
    if len(pos_tags) == 1:
        return is_noun(pos_tags[0])
    return False


if __name__ == '__main__':
    pcfg = Grammar()
    words = [('people', 0.5, 'Noun'), ('fish', 0.2, 'Noun'), ('tanks', 0.2, 'Noun'), ('rods', 0.1, 'Noun'),
             ('people', 0.1, 'Verb'), ('fish', 0.6, 'Verb'), ('tanks', 0.3, 'Verb'), ('with', 1.0, 'Prepositional')]
    np = 'Noun Phrase'
    pp = 'Prepositional Phrase'
    vp = 'Verb Phrase'
    n = 'Noun'
    p = 'Prepositional'
    v = 'Verb'
    pcfg.add_rules(tuple([is_sentence]),
                   tuple([1.0]), name='Sentence', output=[tuple([np, vp])])
    pcfg.add_rules(tuple([is_noun_phrase_a, is_noun_phrase_b, is_noun_phrase_c]),
                   tuple([0.1, 0.2, 0.7]), name='Noun Phrase', output=[tuple([np, np]), tuple([np, pp]), tuple([n])])
    pcfg.add_rules(tuple([is_verb_phrase_a, is_verb_phrase_b]),
                   tuple([0.6, 0.4]), name='Verb Phrase', output=[tuple([v, np, pp]), tuple([v, np])])
    pcfg.add_rules(tuple([is_prepositional_phrase]),
                   tuple([1.0]), name='Prepositional Phrase', output=[tuple([p, np])])
    pcfg.add_terminal_symbol('.')
    for z, p, n in words:
        pcfg.add_non_terminal_symbol(z, p, n)
    print('Rules', pcfg.show_rules())
    print('Rules Tuples', pcfg.show_rule_tuples())
    print('Non terminal words', pcfg.show_non_terminal_words())
    sample = ['people', 'fish', 'tanks', 'with', 'rods']
    pcfg.add_tree_node('Sentence', tuple([1.0]),
                       [tuple(['Noun Phrase', 'Verb Phrase'])])
    pcfg.add_tree_node('Noun Phrase', tuple([0.1, 0.2, 0.7]),
                       [tuple(['Noun Phrase', 'Noun Phrase']),
                        tuple(['Noun Phrase', 'Prepositional Phrase']),
                        tuple(['Noun'])])
    pcfg.add_tree_node('Verb Phrase', tuple([0.6, 0.4]),
                       [tuple(['Verb', 'Noun Phrase']),
                        tuple(['Verb', 'Noun Phrase', 'Prepositional Phrase'])])
    pcfg.add_tree_node('Prepositional Phrase', tuple([1.0]),
                       [tuple(['Prepositional', 'Noun Phrase'])])
    # pcfg.tree_probability(sample)


class Phrase:
    def __init__(self, name: str):
        self.__name = name
        self.__output = []
        self.__weight = []

    @property
    def name(self):
        return self.__name

    def add_output(self, output: tuple[str], weight: float):
        self.__output.append(output)
        self.__weight.append(weight)

    def verify(self):
        return sum(self.__weight) == 1

    def __eq__(self, other):
        if not isinstance(other, Phrase):
            return False
        return other.name == self.__name and self.verify() == other.verify()


if __name__ == '__main__':
    a = Phrase('Sentence')
    b = Phrase('Sentence')
    a.add_output(tuple(['Noun Phrase', 'Verb Phrase']), 0.75)
    b.add_output(tuple(['Noun Phrase', 'Verb Phrase']), 1)
    print(a == b)

