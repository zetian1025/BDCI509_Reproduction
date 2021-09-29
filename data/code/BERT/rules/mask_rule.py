def mask_count(query):
    query = query.lower()
    three_words = ['birthday', 'date', 'born on', 'established on', 'dissoluted on', 'released on', 'formed on',
                   'death', 'died', 'aired on', 'original release', 'broadcasting on', 'broadcast', 'was dissolved on', 'dissolved on [mask]', "dissolution date", 'dissoluted',
                   'aired on', 'original release', 'broadcasting on', 'broadcast', 'established on', 'established in the year', 'formed on [mask]', 'founded on',
                   'founded in the year', 'created in the year [mask]', 'created on [mask]']
    one_words = ['something', 'a type of',
                 'will be likely', 'is part of', 'are likely to']
    year_words = ['in the year']
    temperature = ['°C', '\u00b0C']

    for word in year_words:
        if query.count(word):
            return [5], 0
    for word in three_words:
        if query.count(word):
            return [5], 0
    for word in one_words:
        if query.count(word):
            return [5], 0
    for word in temperature:
        if query.count(word):
            return [2, 2, 1], 0
    return [2, 1, 1, 1], 0


if __name__ == '__main__':
    query = 'The biologist Peter C. G\u00f8tzsche was born on [MASK].'
    mask_count(query)
