import numpy as np


class NonTerminal:
    def __init__(self, start: str, end: list[str], probability: float = 1.0):
        self.__start = start
        self.__end = end
        self.__probability = probability

    @property
    def start(self):
        return self.__start

    @property
    def end(self):
        return self.__end

    @property
    def probability(self):
        return self.__probability

    def __contains__(self, item):
        return item in self.__end

    def is_empty(self):
        return not bool(self.__end)

    def replace(self, old: str, new: str):
        if old in self.__end:
            self.__end[self.__end.index(old)] = new
        return self

    def remove(self, old: str):
        if old in self.__end:
            self.__end.remove(old)
        return self

    def __copy__(self):
        return NonTerminal(self.__start, self.__end)

    def __str__(self):
        return str(self.__start) + ' -> [' + ' '.join(self.__end) + ']'

    def __len__(self):
        return len(self.__end)

    def __eq__(self, other):
        return self.__start == other.start and self.end == other.end

    @probability.setter
    def probability(self, value):
        self.__probability = value


class Terminal:
    def __init__(self, start: str, end: str, probability: float = 1.0):
        self.__start = start
        self.__end = end
        self.__probability = probability

    def replace(self, new: str):
        self.__start = new

    def __contains__(self, item):
        return item == self.__end

    @property
    def start(self):
        return self.__start

    @property
    def probability(self):
        return self.__probability

    @property
    def end(self):
        return self.__end

    def __str__(self):
        return str(self.__start) + ' -> ' + ''.join(self.__end)

    def __eq__(self, other):
        return self.__start == other.start and self.end == other.end


class Grammar:
    def __init__(self):
        self.__terminal_symbols: list[Terminal] = []
        self.__non_terminal_symbols: list[NonTerminal] = []
        self.__normalized = False

    def add_non_terminal_symbol(self, new_symbol: NonTerminal):
        if new_symbol not in self.__non_terminal_symbols:
            self.__non_terminal_symbols.append(new_symbol)

    def add_terminal_symbol(self, new_symbol: Terminal):
        if new_symbol not in self.__terminal_symbols:
            self.__terminal_symbols.append(new_symbol)

    @staticmethod
    def __is_terminal(non_terminal_term: NonTerminal):
        return len(non_terminal_term) == 1

    @staticmethod
    def __print(sequence: list[NonTerminal] or list[Terminal]):
        print(sorted(list(map(str, sequence))))

    def chomsky_normalization(self):
        empty = list(filter(NonTerminal.is_empty, self.__non_terminal_symbols))
        for e in empty:
            self.__non_terminal_symbols.remove(e)
            containing: list[NonTerminal] = list(filter(lambda x: e.start in x, self.__non_terminal_symbols))
            new_non_terminals = []
            old_size = len(self.__non_terminal_symbols)
            for n in containing:
                new = NonTerminal('' + n.start, [] + n.end, 0.1 * n.probability)
                n.probability = 0.9 * n.probability
                new_non_terminals.append(new.remove(e.start))
            for nnts in new_non_terminals:
                if nnts not in self.__non_terminal_symbols:
                    self.add_non_terminal_symbol(nnts)
            assert len(self.__non_terminal_symbols) == len(new_non_terminals) + old_size
        changed = True
        runs = 0
        while changed:
            runs += 1
            terminal_symbols = list(filter(self.__is_terminal, self.__non_terminal_symbols))
            new_terminals = []
            new_non_terminals = []
            skipped = []
            while terminal_symbols:
                ts: NonTerminal = terminal_symbols.pop(0)
                self.__non_terminal_symbols.remove(ts)
                terminal_end_state = [(''.join(ts.end) == __ts.start, __ts.end, __ts.probability)
                                      for __ts in self.__terminal_symbols]
                terminal_end_state_b = [i[1:3] for i in list(filter(lambda x: x[0], terminal_end_state))]
                non_terminal_end_state = list(filter(lambda x: ''.join(ts.end) in x.start, self.__non_terminal_symbols))
                if terminal_end_state_b:
                    # Create New Output
                    for q, q_prob in terminal_end_state_b:
                        start = '' + str(ts.start)
                        new_terminals.append(Terminal(start, q, probability=ts.probability * q_prob))
                elif ts.start != ''.join(ts.end):
                    # Redirect Existing Output(s)
                    for a in non_terminal_end_state:
                        self.add_non_terminal_symbol(
                            NonTerminal('' + ts.start, [] + a.end, probability=ts.probability * a.probability))
                    if not non_terminal_end_state:
                        self.add_non_terminal_symbol(ts)
                        skipped.append(ts)
            for m in skipped:
                terminal_symbols.append(m)
            changed = len(new_terminals) != 0 or len(new_non_terminals) != 0 or len(skipped) != 0
            for nts in new_terminals:
                if nts not in self.__terminal_symbols:
                    self.add_terminal_symbol(nts)
            for nnts in new_non_terminals:
                if nnts not in self.__non_terminal_symbols:
                    self.add_non_terminal_symbol(nnts)

        self.__trim()
        self.__shrink()
        print('Normalized in', runs, 'runs')
        self.__normalized = True

    @property
    def __grammar(self):
        g = {}
        for nts in self.__non_terminal_symbols:
            g[nts.start + ' -> [' + ' '.join(nts.end) + ']'] = nts
        for ts in self.__terminal_symbols:
            g[ts.start + ' -> [' + ts.end + ']'] = ts
        return g

    @property
    def non_terminal_symbols(self) -> list[str]:
        symbol_set = set()
        for i in self.__terminal_symbols:
            symbol_set.add(i.start)
        for i in self.__non_terminal_symbols:
            symbol_set.add(i.start)
            for end in i.end:
                symbol_set.add(end)
        return sorted(list(symbol_set))

    @property
    def __binarization(self):
        symbol_dict = {}
        for i in self.__non_terminal_symbols:
            symbol_dict[tuple(i.end)] = i
        return symbol_dict

    @property
    def __unaries(self):
        symbol_dict = {}
        for i in self.__terminal_symbols:
            symbol_dict[tuple(i.end)] = i
        return symbol_dict

    @property
    def terminal_symbols(self):
        symbol_set = set()
        for i in self.__terminal_symbols:
            symbol_set.add(i.end)
        return sorted(list(symbol_set))

    def __trim(self):
        non_terminal_symbols = set()
        for non_terminal_symbol in self.__non_terminal_symbols:
            non_terminal_symbols.add(non_terminal_symbol.start)
            for i in non_terminal_symbol.end:
                non_terminal_symbols.add(i)
        dropped = list(filter(lambda x: x,
                              [i if i.start not in non_terminal_symbols else None for i in self.__terminal_symbols]))
        for i in dropped:
            self.__terminal_symbols.remove(i)

    def __shrink(self):
        too_large_sets = list(filter(lambda x: len(x.end) > 2, self.__non_terminal_symbols))
        self.__print(too_large_sets)
        for a in too_large_sets:
            self.__non_terminal_symbols.remove(a)
            extras = [a.end[i] for i in range(len(a.end) - 2)]
            name: str = a.start
            for i in extras:
                _end_: str = '@' + name + '_' + i
                new_base = NonTerminal(name, [i, _end_])
                name = _end_
                self.__non_terminal_symbols.append(new_base)
            new_base = NonTerminal(name, a.end[-2:])
            self.__non_terminal_symbols.append(new_base)

    def get_non_terminal(self, symbol: str):
        return list(filter(lambda x: x.start == symbol, self.__non_terminal_symbols))

    def get_terminal(self, symbol: str):
        return list(filter(lambda x: x.start == symbol, self.__terminal_symbols))

    def binary_config(self, left_type: str, left_prob: float, right_type: str, right_prob: float):
        guess = tuple([left_type, right_type])
        if guess in self.__binarization:
            nst: NonTerminal = self.__binarization[guess]
            return nst, nst.probability * left_prob * right_prob
        return None, 0

    def unary_config(self, left_type: str, left_prob: float):
        if left_type in self.__unaries:
            st: Terminal = self.__unaries[tuple([left_type])]
            return st, st.probability * left_prob
        return None, 0

    def cky_parsing(self, sentence: list[str]):
        details = {(i, j): {} for i in range(len(sentence)) for j in range(len(sentence) - i)}
        for j in range(len(sentence)):
            details[(0, j)] = {}
            for ts in self.__terminal_symbols:
                if ts.end == sentence[j]:
                    details[(0, j)][ts.start] = ts.probability
        for i in range(1, len(sentence)):  # Levels of pyramid
            for j in range(len(sentence) - i):  # Rows in current level
                details[(i, j)] = {}
                for m in details[(i - 1, j)]:
                    for n in details[(i - 1, j + 1)]:
                        left_prob = details[(i - 1, j)][m]
                        right_prob = details[(i - 1, j + 1)][n]
                        trial, prob = self.binary_config(m, left_prob, n, right_prob)
                        if trial:
                            if trial.start in details[(i, j)]:
                                details[(i, j)][trial.start] = max([prob, details[(i, j)][trial.start]])
                            else:
                                details[(i, j)][trial.start] = prob
        print(details)
        return details[(len(sentence) - 1, 0)]

    def cky_2(self, words):
        score = np.zeros((len(words) + 1, len(words) + 1, len(self.non_terminal_symbols)))
        back: list[list[list]] = []
        for i in range(len(words) + 1):
            back_i = []
            for j in range(len(words) + 1):
                back_i.append([None for _ in range(len(self.non_terminal_symbols))])
            back.append(back_i)
        # back = [[[] * len(self.non_terminal_symbols)] * (len(words) + 1)] * (len(words) + 1)
        print(back, len(self.non_terminal_symbols))
        print(np.asarray(back).shape, score.shape)
        # Lexicon
        for i in range(len(words)):
            for t in range(len(self.non_terminal_symbols)):
                equation = self.non_terminal_symbols[t] + ' -> [' + words[i] + ']'
                if equation in self.__grammar:
                    score[i, i + 1, t] = self.__terminal_symbols[t].probability
            # Handle Unaries
            added = True
            while added:
                added = False
                for a in range(len(self.non_terminal_symbols)):
                    for b in range(len(self.non_terminal_symbols)):
                        equation = self.non_terminal_symbols[a] + ' -> [' + self.non_terminal_symbols[b] + ']'
                        if score[i, i + 1, b] > 0 and equation in self.__grammar:
                            prob = self.__grammar[equation].probability * score[i, i + 1, b]
                            if prob > score[i, i + 1, a]:
                                score[i, i + 1, a] = prob
                                back[i][i + 1][a] = b
                                added = True
        # Rest of the chart
        for span in range(2, len(words)):
            for begin in range(len(words) - span):
                end = begin + span
                for split in range(begin + 1, end):
                    for a in range(len(self.non_terminal_symbols)):
                        for b in range(len(self.non_terminal_symbols)):
                            for c in range(len(self.non_terminal_symbols)):
                                equation = self.non_terminal_symbols[a] + ' -> [' + self.non_terminal_symbols[b] + ' ' + \
                                           self.non_terminal_symbols[c] + ']'
                                if equation in self.__grammar:
                                    prob = score[begin, split, b] * score[split, end, c] * self.__grammar[
                                        equation].probability
                                    if prob > score[begin, end, a]:
                                        score[begin, end, a] = prob
                                        back[begin][end][a] = (split, b, c)
                # Handle Unaries
                added = True
                while added:
                    added = False
                    for a in range(len(self.non_terminal_symbols)):
                        for b in range(len(self.non_terminal_symbols)):
                            equation = self.non_terminal_symbols[a] + ' -> [' + self.non_terminal_symbols[b] + ']'
                            if equation in self.__grammar:
                                prob = self.__grammar[equation].probability * score[begin, end, b]
                                if prob > score[begin, end, a]:
                                    score[begin, end, a] = prob
                                    back[begin][end][a] = b
                                    added = True
        print(back)
        return score


if __name__ == '__main__':
    grammar = Grammar()
    terminal_symbol_set = [Terminal('N', 'people', 0.5), Terminal('N', 'fish', 0.2), Terminal('N', 'tanks', 0.2),
                           Terminal('N', 'rods', 0.1), Terminal('V', 'people', 0.1), Terminal('V', 'fish', 0.6),
                           Terminal('V', 'tanks', 0.3), Terminal('P', 'with', 1.0)]
    non_terminal_symbol_set = [NonTerminal('S', ['NP', 'VP'], 1.0), NonTerminal('VP', ['V', 'NP'], 0.6),
                               NonTerminal('VP', ['V', 'NP', 'PP'], 0.4), NonTerminal('NP', ['NP', 'NP'], 0.1),
                               NonTerminal('NP', ['NP', 'PP'], 0.2), NonTerminal('NP', ['N'], 0.65),
                               NonTerminal('NP', [], 0.05), NonTerminal('PP', ['P', 'NP'], 1.0)]
    for s in terminal_symbol_set:
        grammar.add_terminal_symbol(s)
    for s in non_terminal_symbol_set:
        grammar.add_non_terminal_symbol(s)
    grammar.chomsky_normalization()
    for term in ['S', 'NP', 'VP', 'PP', 'P', 'V', 'N']:
        print('Non-Terminal:', list(map(str, grammar.get_non_terminal(term))))
        print('Terminal:', list(map(str, grammar.get_terminal(term))))
    print(grammar.terminal_symbols)
    print(grammar.non_terminal_symbols)
    print(grammar.cky_parsing(['fish', 'people', 'fish', 'tanks']))
    # print(grammar.cky_2(['fish', 'people', 'fish', 'tanks']))

    grammar2 = Grammar()
    non_terminal_symbol_set2 = [NonTerminal('S', ['NP', 'VP'], 0.9), NonTerminal('S', ['VP'], 0.1),
                                NonTerminal('VP', ['V', 'NP'], 0.5), NonTerminal('VP', ['V'], 0.1),
                                NonTerminal('VP', ['V', '@VP_V'], 0.3), NonTerminal('VP', ['V', 'PP'], 0.1),
                                NonTerminal('@VP_V', ['NP', 'PP'], 1.0), NonTerminal('NP', ['NP', 'NP'], 0.1),
                                NonTerminal('NP', ['NP', 'PP'], 0.2), NonTerminal('NP', ['N'], 0.7),
                                NonTerminal('PP', ['P', 'NP'], 1.0)]
    for s in terminal_symbol_set:
        grammar2.add_terminal_symbol(s)
    for s in non_terminal_symbol_set2:
        grammar2.add_non_terminal_symbol(s)
    print(grammar2.cky_2(['fish', 'people', 'fish', 'tanks']))
