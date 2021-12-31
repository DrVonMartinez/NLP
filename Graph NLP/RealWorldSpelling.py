import networkx as nx
from Spelling_Correction import one_step_spelling


def noisy_channel(noisy_sentence: str):
    words = noisy_sentence.split(' ')
    graph = nx.Graph(name='Noisy').to_directed()
    prior_words = []
    for word in words:
        alternate_spellings = one_step_spelling(word)
        print(alternate_spellings)
        for alternate_spelling in alternate_spellings:
            graph.add_node(alternate_spelling)
            for prior_word in prior_words:
                graph.add_edge(prior_word, alternate_spelling)
        prior_words = alternate_spellings
    print(nx.adjacency_matrix(graph))


noisy_channel('two of thew')
