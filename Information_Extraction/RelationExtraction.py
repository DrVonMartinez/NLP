# Extracting relation triples
# Founding-year (IBM, 1911)
# Founding-location (IBM, New York)

# Support Question Answering:
# (acted-in ?x _Movie_) (is-a ?y actor) (granddaughter-of ?x ?y)

# Automated Content Extraction (ACE)
import nltk

ace: dict[str, list[str]] = {"Person-Social": ['Family', 'Business', 'Lasting Personal'],
                             "Physical": ["Located", "Near"],
                             "General Affiliation": ['Citizen-Resident-Ethnicity-Religion', 'Org-Location-Origin'],
                             "Part-Whole": ['Subsidiary', 'Geographical'],
                             "Organization Affiliation": ['Founder', 'Ownership', 'Membership', 'Sports-Affiliation',
                                                          'Employment', 'Student-Alum', 'Investor'],
                             "Artifact": ['User-Owner-Inventor-Manufacturer']}
# UMLS: Unified Medical Language System
# Resource Description Framework (RDF)
# subject predicate object

# Ontological Relations
# Is-a (hypernym)
#   Instance-of

def is_a(x: str, source: str):
    """
    Hearst, 1992: Automatic Acquisition of Hyponyms
    :param x:
    :param source:
    :return:
    """
    if x not in source:
        raise ValueError(x + ' must be in the data source for this level of relationship')
    split_source = source.split(' ')
    pos_tags = nltk.pos_tag(split_source)
    print(pos_tags)
    description = []

    def pre_chaining(word: str):
        index = split_source.index(word)
        chain = True
        i = index
        while i > 0 and chain:
            chain = False
            if pos_tags[i][1] in ['JJ', 'RB']:
                chain = True
            i -= 1
        return split_source[i: index]

    def post_chaining(word: str):
        index = split_source.index(word)
        chain = True
        i = index
        while i < len(pos_tags) and chain:
            chain = False
            if pos_tags[i][1] in ['JJ', 'RB']:
                chain = True
            i += 1
        return split_source[index + 1: i]

    if 'such as' in source:
        description = pre_chaining('such')
    elif ', ' + x in source:
        description = pre_chaining(x)
    elif ', and ' + x in source:
        description = pre_chaining('and')
    elif ', or ' + x in source:
        description = pre_chaining('or')
    elif 'including ' + x in source:
        description = pre_chaining('including')
    elif ', especially ' + x in source:
        description = pre_chaining('especially')
    elif x + ' or other' in source or x + ' and other' in source:
        description = post_chaining('other')
    return ' '.join(description).replace(',', '')


if __name__ == '__main__':
    test_source = 'Agar is a substance prepared from a mixture of red algae, such as Gelidium, ' \
                  'for laboratory or industrial use'
    print(is_a('Gelidium', test_source))
    test_source2 = 'Mice and other rodents'
    print(is_a('Mice', test_source2))
    # Intuition ->
    #   located-in(organization,location)
    #   founded(person,organization)
    #   cures(drug,disease) 
