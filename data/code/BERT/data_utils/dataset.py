from torch.utils import data
import json
import string


def read_inputs(input_file):
    with open(input_file) as f:
        lines = f.readlines()
        data = [json.loads(l.strip()) for l in lines]
    return data


class KCTDataset(data.Dataset):
    def __init__(self, data_type, data):
        super().__init__()
        self.data_type = data_type
        self.sentences, self.labels, self.qid = [], [], []
        self.exclude = set(string.punctuation)

        if self.data_type == 'test':
            for d in data:
                self.sentences.append(self.__stmp__(
                    d['decoration1']) + " [SEP] " + d['query'])
                self.labels.append(' ')
                self.qid.append(d['qid'])
                self.sentences.append(self.__stmp__(
                    d['decoration2']) + " [SEP] " + d['query'])
                self.labels.append(' ')
                self.qid.append(d['qid'])

        else:
            for d in data:
                self.sentences.append(self.__stmp__(
                    d['decoration1']) + " [SEP] " + d['query'])
                self.labels.append(self.__tmp__(d['answer'][0]))
                self.qid.append(d['qid'])

                self.sentences.append(self.__stmp__(
                    d['decoration2']) + " [SEP] " + d['query'])
                self.labels.append(self.__tmp__(d['answer'][0]))
                self.qid.append(d['qid'])

    def __stmp__(self, query):
        query_list = query.split()
        for month in ['January', 'February', 'March', 'April', 'May', 'June', 'July', 'August', 'September', 'October', 'November', 'December']:
            if month in query_list:
                month_index = query_list.index(month)
                query = query.replace(' '+query_list[month_index-1], '')
                query = query.replace(query_list[month_index+1]+' ', '')
                return query
        return query

    def __tmp__(self, answer):
        answer_list = answer.split()
        if len(answer_list) == 3:
            if answer_list[1] in ['January', 'February', 'March', 'April', 'May', 'June', 'July', 'August', 'September', 'October', 'November', 'December']:
                return answer_list[1]
        return answer

    def __len__(self):
        return len(self.sentences)

    def __getitem__(self, i):
        return self.sentences[i], self.labels[i], self.qid[i]
