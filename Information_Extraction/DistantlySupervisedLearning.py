import types


class DistantlySupervisedLearning:
    def __init__(self):
        self.__relations = set()
        self.__tuple_relations: dict[str, list[tuple]] = {}
        self.__relation_features: dict[str, set[str]] = {}

    def add_relation(self, relation: object):
        if callable(relation):
            self.__tuple_relations[str(relation.__name__)] = []
            self.__relation_features[str(relation.__name__)] = set()
            self.__relations.add(relation)
        else:
            raise TypeError(str(relation) + ' should be a callable function')

    def add_tuple(self, relation: object, sequence: tuple):
        if relation not in self.__relations:
            self.add_relation(relation)
        self.__tuple_relations[str(relation.__name__)].append(sequence)

    def show_relations(self):
        print('\n'.join(list(self.__tuple_relations.keys())))

    @staticmethod
    def __drop_digit(u: str) -> str:
        if u.isdigit():
            return 'X'
        return u

    def parse_frequent_features(self, relation_name: str, text: list[str]):
        if relation_name in list(self.__tuple_relations.keys()):
            relation = list(filter(lambda x: x.__name__ == relation_name, self.__relations))[0]
            for line in text:
                for t in self.__tuple_relations[relation_name]:
                    tuple_types: tuple = relation(t)
                    if sum(map(lambda u: u[1] in line, tuple_types)) >= len(t):
                        new_line = line
                        for c in range(len(tuple_types)):
                            new_line = new_line.replace(str(tuple_types[c][1]), str(tuple_types[c][0]))
                        self.__relation_features[relation_name].add(''.join(map(self.__drop_digit, new_line)))
        else:
            raise KeyError(relation_name + ' is not defined')

    def show_features(self, relation_name: str):
        if relation_name in list(self.__tuple_relations.keys()):
            print('\n'.join(self.__relation_features[relation_name]))
        else:
            raise KeyError(relation_name + ' is not defined')


def born_in(x: tuple[str, str]) -> tuple:
    name, location = x
    first_name, last_name = name.split(':')
    return tuple([('PERSON', name), ('PERSON', first_name), ('PERSON', last_name), ('LOCATION', x[1])])


if __name__ == '__main__':
    sample_relation = born_in
    sample_tuples = [('Edwin:Hubble', 'Marshfield'), ('Albert:Einstein', 'Ulm')]
    sample_text = ['Hubble was born in Marshfield',
                   'Hubble, born (1889), Marshfield',
                   'Einstein, born (1879), Ulm',
                   "Hubble's birthplace in Marshfield"]
    dsl = DistantlySupervisedLearning()
    dsl.add_relation(sample_relation)
    dsl.show_relations()
    for st in sample_tuples:
        dsl.add_tuple(sample_relation, st)
    dsl.parse_frequent_features(born_in.__name__, sample_text)
    dsl.show_features(born_in.__name__)
