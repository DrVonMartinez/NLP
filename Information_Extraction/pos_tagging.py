from nltk.corpus import brown


def feature_the(word: str):
    if word == 'the':
        return 'DT'


def __word_shape(word: str):
    def convert(char: str):
        if str(char).islower():
            return 'x'
        elif str(char).isupper():
            return 'X'
        elif str(char).isdigit():
            return 'd'
        else:
            return str(char)

    new_word = ''
    if len(word) <= 4:
        new_word = ''.join(map(convert, word))
    else:
        new_word += ''.join(map(convert, word[0:2]))
        new_word += ''.join(list(set(map(convert, word[2:-2]))))
        new_word += ''.join(map(convert, word[-2:]))
    return new_word


def prefix_un(word: str):
    if word.lower().startswith('un'):
        return 'JJ'


def suffix_ly(word: str):
    if word.lower().endswith('ly'):
        return 'RB'


def capitalization(word: str):
    if word[0].isupper():
        return 'NNP'


def word_shapes_adj(word: str):
    shape = __word_shape(word)
    for p in ['xx', 'dd']:
        while p in shape:
            shape.replace(p, p[0])
    if shape == 'd-x':
        return 'JJ'


if __name__ == '__main__':
    words = brown.words()
    print(words)
    print(__word_shape('the'))
    print(__word_shape('there'))
    print(__word_shape('THE'))
    print(__word_shape('THERE'))
    print(__word_shape('TH3RE'))
    print(__word_shape('TH-dRE'))
    print(__word_shape('TH-d3RE'))
