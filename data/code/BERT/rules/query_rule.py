
def name_establishment(query, target_word):
    query = query.replace('formed', target_word)
    query = query.replace('originally founded', target_word)
    return query

def plural2singular(query):
    query = query.replace('screenwriters', 'screenwriter')
    query = query.replace('are [', 'is [')
    query = query.replace('symptoms', 'symptom')
    return query

def query_cleaning(query):
    query = name_establishment(query, 'established')
    query = plural2singular(query)
    return query