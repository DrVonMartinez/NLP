# Reasons:
#   Machine Translation
#   Spell Correction
#   Speech Recognition
import numpy as np


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
    _n_ = n - 1
    training = training_text.replace('.', '</s> <s>').split(' ')
    gram: dict[str] = {}
    word_count: dict[str] = {training[0]: 1}
    for i in range(1, len(training)):
        print(' '.join(training[i - _n_: i + 1]))
        if ' '.join(training[i - _n_: i + 1]) in gram.keys():
            gram[' '.join(training[i - _n_: i + 1])] += 1
        else:
            gram[' '.join(training[i - _n_: i + 1])] = 1
        if training[i] in word_count.keys():
            word_count[training[i]] += 1
        else:
            word_count[training[i]] = 1

    def n_set(stream: str) -> list[str]:
        temp = stream.split(' ')
        return [' '.join(temp[j: j + n]) for j in range(len(temp) - n)]

    def n_phrase(phrase):
        return np.log(gram[phrase] / word_count[phrase.split(' ')[-1 * n]])

    return np.exp(sum(map(n_phrase, n_set(sentence))))


print(n_gram('<s> I am Sam </s>', "<s> I am Sam </s> <s> Sam I am </s> <s> I do not like green eggs and ham </s>", 2))


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
