def word_contains_number(word: str) -> bool:
    return any(map(lambda x: str(x).isnumeric(), word))


def is_floating_point_number(word: str) -> bool:
    try:
        return str(float(word)) == word
    except ValueError:
        pass
    return False


def word_ends_with(word: str, suffix: str) -> bool:
    return word[-len(suffix):] == suffix


def f1_function(x: str):
    x, y = x.split(' ')
    return x == 'in' and str(y[0]).isupper()


def f2_function(x: str):
    x, y = x.split(' ')
    return any(list(map(lambda z: z in ['á', 'é', 'í', 'ó', 'ú'], y)))


def f3_function(x: str):
    return x[-1] == 'c'


def is_capital(x: str):
    x, y = x.split(' ')
    return y[0].isupper()


if __name__ == '__main__':
    print(word_contains_number('5'))
    print(word_contains_number('3.6'))
    print(word_contains_number('3a'))
    print(word_ends_with('running', 'ing'))
    print(word_ends_with('jump', 'ing'))
    print(word_ends_with('3.3', '3'))
