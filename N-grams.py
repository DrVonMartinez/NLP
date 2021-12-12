# Reasons:
#   Machine Translation
#   Spell Correction
#   Speech Recognition
import numpy as np
import numpy as np
from scipy.optimize import curve_fit

sample_sentence = "<s> I am Sam </s> <s> Sam I am </s> <s> I do not like green eggs and ham </s>"
test_sentence = '<s> I am Sam </s>'
test_sentence2 = '<s> I am the Sam </s>'
test_sentence3 = '<s> I am'
sample_sentence2 = 'Sam I am I am Sam I do not eat'


def powlaw(x, a, b):
    # Power law per stack overflow
    # https://stackoverflow.com/questions/41109122/fitting-a-curve-to-a-power-law-distribution-with-curve-fit-does-not-work
    return a * np.power(x, b)


def simple_training(training_text: str, n) -> tuple[dict, dict]:
    _n_ = n - 1
    training = training_text.split(' ')
    gram: dict[str] = {}
    word_count: dict[str] = {}
    for i in range(_n_, len(training)):
        gram.setdefault(' '.join(training[i - _n_: i + 1]), 0)
        gram[' '.join(training[i - _n_: i + 1])] += 1
    for i in range(len(training)):
        word_count.setdefault(training[i], 0)
        word_count[training[i]] += 1
    return word_count, gram


def n_gram(sentence: str, training_text: str, n: int):
    """
    P(W) or P(wn | w1 w2 w3 ...  wn-1)
    :param sentence
    :param training_text
    :param n
    :return:
    """
    # Chain Rule:
    #   P(A|B) = P(A,B) / P(B)
    #   P(A|B) P(B) = P(A,B)
    #   P(A,B) = P(A|B) P(B)
    # P(w1 w2 w3 ... wk) ≈ Π P(wi | wi-n ... wi-1)
    word_count, gram = simple_training(training_text, n)

    def n_set(stream: str) -> list[str]:
        temp = stream.split(' ')
        return [' '.join(temp[j: j + n]) for j in range(len(temp) - n)]

    def n_phrase(phrase):
        try:
            return np.log(gram[phrase] / word_count[phrase.split(' ')[-1 * n]])
        except KeyError:
            return np.log(0)

    return np.exp(sum(map(n_phrase, n_set(sentence))))


def simple_interpolation(sentence: str, training_set: str):
    a: float = 1 / 3
    b: float = 1 / 3
    c: float = 1 / 3
    unigram = a * n_gram(sentence, training_set, 1)
    bigram = b * n_gram(sentence, training_set, 2)
    trigram = c * n_gram(sentence, training_set, 3)
    assert a + b + c == 1
    return unigram + bigram + trigram


def contextual_interpolation(sentence: str, training_set: str):
    a = {}
    b = {}
    c = {}
    unigram = a[sentence.split(' ')[-2:]] * n_gram(sentence, training_set, 1)
    bigram = b[sentence.split(' ')[-2:]] * n_gram(sentence, training_set, 2)
    trigram = c[sentence.split(' ')[-2:]] * n_gram(sentence, training_set, 3)
    assert a + b + c == 1
    return unigram + bigram + trigram


def unigram_prior(sentence: str, training_text: str, k: int):
    word_count, gram = simple_training(training_text, 1)
    v = len(word_count.values())

    def n_set(stream: str) -> list[str]:
        temp = stream.split(' ')
        return [' '.join(temp[j: j + 1]) for j in range(len(temp) - 1)]

    def n_phrase(phrase):
        try:
            return np.log((gram[phrase] + k) / (word_count[phrase.split(' ')[-1]] + k * v))
        except KeyError:
            return np.log(1 / v)

    return np.exp(sum(map(n_phrase, n_set(sentence))))


# print(test_sentence2)
# print(n_gram(test_sentence2, sample_sentence, 1))
# print(unigram_prior(test_sentence2, sample_sentence, 0))


def good_turing(sentence: str, training_text: str):
    n_c = {}  # count of things we've seen c times
    word_count, gram = simple_training(training_text, 1)
    max_count = max(word_count.values())
    for i in range(max_count):
        count = len(list(filter(lambda x: x == i + 1, word_count.values())))
        if count > 0:
            n_c['N_' + str(i + 1)] = count

    v = sum(word_count.values())
    _n_c_ = {}
    for i in range(max_count):
        t = n_c.copy()
        _n_c_['N_' + str(i)] = (((i + 1) * t.setdefault('N_' + str(i + 1), 0)) / t.setdefault('N_' + str(i), 1)) / v
    xdata = [j for j in range(max_count + 2)]
    ydata = [_n_c_.setdefault('N_' + str(j), 0) for j in range(max_count + 2)]
    y__data, _ = curve_fit(powlaw, xdata, ydata)
    __n_c__ = {"N_" + str(j): powlaw(xdata[j - 3], *list(y__data)) for j in range(4, max_count + 2)}
    __n_c__["N_0"] = _n_c_['N_0']
    __n_c__["N_1"] = _n_c_['N_1']
    __n_c__["N_2"] = _n_c_['N_2']
    __n_c__["N_3"] = _n_c_['N_3']
    print(__n_c__)

    def n_set(stream: str) -> list[str]:
        temp = stream.split(' ')
        return [' '.join(temp[j: j + 1]) for j in range(len(temp))]

    def n_phrase(phrase, log=True):
        try:
            n_c.setdefault("N_" + str(gram[phrase] + 1), 1)
            x = (gram[phrase] + 1) * n_c["N_" + str(gram[phrase] + 1)] / n_c["N_" + str(gram[phrase])]
        except KeyError:
            x = n_c["N_1"]
        if log:
            return np.log(x / v)
        return x / v

    def n_phrase2(phrase, log=True):
        if log:
            return np.log(__n_c__["N_" + str(gram.setdefault(phrase, 0))])
        return __n_c__["N_" + str(gram[phrase])]

    return np.exp(sum(map(n_phrase, n_set(sentence))))


fish_test = ['carp' for i in range(10)] + ['perch', 'perch', 'perch', 'whitefish', 'whitefish', 'trout', 'salmon',
                                           'eel']
fish_test_str = ' '.join(fish_test)


# print('Good Turing:', a := good_turing('carp', fish_test_str))
# print('Good Turing:', b := good_turing('perch', fish_test_str))
# print('Good Turing:', c := good_turing('whitefish', fish_test_str))
# print('Good Turing:', d := good_turing('trout', fish_test_str))
# print('Good Turing:', e := good_turing('bass', fish_test_str))


def discount_bigram(sentence: str, training_text: str, discount=0.75):
    word_count, gram = simple_training(training_text, 2)

    def n_set(stream: str) -> list[str]:
        temp = stream.split(' ')
        return [' '.join(temp[j: j + 2]) for j in range(len(temp) - 2)]

    def continuation(word: list[str]):
        continuation_dict = {}
        for i in gram:
            word__, word_i = i.split(' ')
            if word_i == word[-1]:
                continuation_dict.setdefault(word__, 0)
                continuation_dict[word__] += 1
        return len(continuation_dict) / len(gram)

    def interpolation_weight(word: list[str]):
        num_word_count, num_word_gram = simple_training(training_text, len(word))
        interpolation_dict = {}
        for i in num_word_gram:
            split = i.split(' ')
            words, word_i = split[:-1], split[-1]
            if word_i == word[-1]:
                interpolation_dict.setdefault(word_i, 0)
                interpolation_dict[word_i] += 1
        print(interpolation_dict)
        return discount * len(interpolation_dict) / num_word_count[word[-1]]

    def n_phrase(phrase):
        words = phrase.split(' ')
        theta = interpolation_weight(words[-2:]) * continuation(words[-1:])
        print(interpolation_weight(words[-2:]), words[-2:])
        print(continuation(words[-1:]), words[-1:])
        try:
            x = (gram[phrase] - discount) / sum(word_count.values())
        except KeyError:
            x = 0
        print(max(x, 0) + theta)
        return np.log(max(x, 0) + theta)

    return np.exp(sum(map(n_phrase, n_set(sentence))))


def higher_order_kneser_ney(sentence, training_text, n, discount=0.75):
    word_count, gram = simple_training(training_text, n)

    def n_set(stream: str) -> list[str]:
        temp = stream.split(' ')
        return [' '.join(temp[j: j + n]) for j in range(len(temp) - n)]

    def continuation(word):
        continuation_dict = {}
        if len(word) == 2:
            num_word_count, num_word_gram = simple_training(training_text, len(word))
            for i in num_word_gram:
                split: list[str] = i.split(' ')
                words, word_i = split[:-1], split[-1]
                if word_i == word[-1]:
                    continuation_dict.setdefault(' '.join(words), 0)
                    continuation_dict[' '.join(words)] += 1
            return len(continuation_dict) / len(num_word_gram)
        else:
            return n_phrase(' '.join(word[1:]), len(word) - 1)

    def interpolation_weight(word):
        num_word_count, num_word_gram = simple_training(training_text, len(word))
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
        num_word_count, _ = simple_training(training_text, num)
        theta = interpolation_weight(words[-1 * num:]) * continuation(words[-1 * (num + 1):])
        print(interpolation_weight(words[-1 * num:]), words[-1 * num:])
        try:
            x = (gram[phrase] - discount) / sum(num_word_count.values())
        except KeyError:
            x = 0
        print(max(x, 0) + theta)
        return np.log(max(x, 0) + theta)
    result = n_set(sentence)
    return np.exp(sum(map(n_phrase, result, [n] * len(result))))


print('DB:', discount_bigram(test_sentence3, sample_sentence))
print('HOKN:', higher_order_kneser_ney(test_sentence3, sample_sentence, 2))
