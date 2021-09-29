import argparse
import json


def read_inputs(input_file):
    with open(input_file) as f:
        lines = f.readlines()
        data = [json.loads(l.strip()) for l in lines]
    return data


def keyword_select(data, args):
    input_data = data['query']
    if args.data_type == 'train':
        if data['domain'] == 'Common Sense':
            return 'CommonSense'
    for key_word in args.key_words:
        for word in args.key_words.get(key_word):
            if input_data.count(word):
                return key_word
    return 'CommonSense'


def init_argparse():
    parser = argparse.ArgumentParser()
    parser.add_argument('--read_path', type=str,
                        default='./data/KCT_test_public.txt')
    parser.add_argument('--write_path', type=str,
                        default='./data/self_made/test2/')
    parser.add_argument('--data_type', type=str, default='test')
    args = parser.parse_args()

    args.key_words = {
        'useless': ['seasons', 'period ', "a song by", 'an album by', 'buried at', 'music genre', 'released on', 'released date'],
        'date_birth_and_death': ['date of birth', 'birthday', 'born on [MASK]', 'birth date', 'birth is [MASK]', 'born on date [MASK]', 'born in the year', ' death ', ' died '],
        'date_dissolution': ['dissolved on [MASK]', "dissolution date", 'dissoluted'],
        'date_establishment': ['established on', 'established in the year', 'formed on [MASK]', 'founded on', 'founded in the year', 'created in the year [MASK]', 'created on [MASK]'],
        'date_broadcast': ['aired on', 'Original release', 'broadcasting on', 'broadcast'],

        'number_MeltingPoint': ['melting point'],

        'disease_symptoms': ['symptoms'],
        'disease_cause': ['disease', 'endemic'],

        'name_theme': ['theme'],

        'place_birth': ['born at', 'born in', 'from the country [MASK]', 'nationality'],
        'place_native': ['produced in', 'made by the country', 'native', 'formed', 'fromed', 'based in', 'founded', 'legislative act', 'a [MASK] film', 'law act', 'produced by', 'equipment made by', 'from country', 'owned by the country', 'belongs to', 'the country [MASK]', 'in country [MASK]'],
        'place_capital': ['capital'],
        'place_university': ['a degree', 'university', 'graduated from'],
        'place_found': ['was found in'],
        'place_FilmingCountry': ['filming country'],

        'name_manufacturer': ['manufacturer'],

        'name_war': [' war ', 'conflict'],
        'name_TV': ['series'],

        'person_parents': ['father is', 'the father of', 'father.', 'mother is', 'the mother of'],
        'person_film': ['directed', 'fiml starring', 'film starring', 'fiml stars', 'cast member', 'written', 'screenwriters', 'wrote', 'screenwriter'],
        'person_supervisor': ['supervised', 'supervisor', 'advisor'],

        'CommonSense': ['common sense']
    }

    return args


if __name__ == '__main__':

    args = init_argparse()

    dataset = read_inputs(args.read_path)
    labels = [keyword_select(data, args) for data in dataset]

    for key_word in args.key_words:
        with open(args.write_path + key_word + '.txt', 'w+') as f:
            for data, label in zip(dataset, labels):
                if label == key_word:

                    f.writelines(json.dumps(data) + '\n')
