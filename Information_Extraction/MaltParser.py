class MaltParser:
    def __init__(self):
        self.__sigma: list[str] = ["ROOT"]
        self.__buffer: list[str] = []
        self.__dependency_arcs = []

    @property
    def buffer(self):
        return self.__buffer

    @buffer.setter
    def buffer(self, words: list[str]):
        self.__buffer = words

    def _shift(self, buffer):
        """
        sigma, w_i | B, A -> sigma | w_i, B, A
        :return:
        """
        self.__sigma.append(buffer.pop(0))
        return buffer

    def _left_arc(self):
        """
        sigma | w_i, w_j | B, A -> sigma, w_j | B, A U {r(w_j, w_i)}
        r'(w_k, w_i) not in A, w_i != ROOT
        :return:
        """
        pass

    def _right_arc(self):
        """
        sigma | w_i, w_j | B, A -> sigma|w_i|w_j, w_i | B, A U {r(w_i, w_j)}
        :return:
        """
        pass

    def _reduce(self):
        """
        sigma | w_i, B, A -> sigma, B, A
        r'(w_k', w_i) in A
        :return:
        """

    def dependency_parser(self, buffer: list[str] = None):
        if buffer:
            self.__buffer = buffer
        elif not self.__buffer:
            raise ValueError('There must be a stored buffer')
        buffer = self.__buffer
        while buffer:
            print(self.__sigma, buffer, self.__dependency_arcs)
            buffer = self._shift(buffer)


if __name__ == '__main__':
    sentence = ['Happy', 'children', 'like', 'to', 'play', 'with', 'their', 'friends']
    grammar = MaltParser()
    grammar.buffer = sentence
    grammar.dependency_parser()
