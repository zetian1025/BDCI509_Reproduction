import argparse
import json
import torch
import random
from random import randint
from torch.utils import data
from tqdm import tqdm
from sentence_transformers import SentenceTransformer, util
import os


def read_inputs(input_file):
    with open(input_file) as f:
        lines = f.readlines()
        data = [json.loads(l.strip()) for l in lines]
    return data


class train_dataset(data.Dataset):
    def __init__(self, train_path):
        read_data = read_inputs(train_path)
        self.data = []
        for line_data in tqdm(read_data):
            query_text = line_data['query']
            answer_text = line_data['answer']
            self.data.append(query_text.replace('[MASK]', answer_text[0]))

        self.show_example()

    def show_example(self):
        idx = randint(0, len(self.data) - 1)
        print("query: ", self.data[idx])
        print("-" * 100)

    def __getitem__(self, item):
        return self.data[item]

    def __len__(self):
        return len(self.data)


class test_dataset(data.Dataset):
    def __init__(self, test_path):
        read_data = read_inputs(test_path)
        self.data = []
        for line_data in tqdm(read_data):
            self.data.append({
                'query': line_data['query'],
                'qid': line_data['qid'],
                'answer': line_data['answer'] if "train" in test_path else "",
            })

        self.show_example()

    def show_example(self):
        idx = randint(0, len(self.data) - 1)
        print("query: ", self.data[idx]['query'])
        print("qid: ", self.data[idx]['qid'])
        print("-" * 100)

    def __getitem__(self, item):
        return self.data[item]

    def __len__(self):
        return len(self.data)


if __name__ == '__main__':
    parser = argparse.ArgumentParser()

    parser.add_argument("--pretrained_model", type=str, default="./models/paraphrase-mpnet-base-v2",
                        help="pretrained model name or path to local checkpoint")
    parser.add_argument("--train_set", type=str, default="./data/self_made/train2/",
                        help="Input dataset in json format.")
    parser.add_argument("--test_set", type=str, default="./data/self_made/train2/",
                        help="Input dataset in json format")
    parser.add_argument("--output_dir", type=str,
                        default="./data/self_made/suffix2/train/", help="Output dir.")

    args = parser.parse_args()

    device = "cuda" if torch.cuda.is_available() else "cpu"
    model = SentenceTransformer(args.pretrained_model)

    set_list = os.listdir(args.train_set)

    results = []
    all_len = 0
    for set in set_list:
        if len(read_inputs(args.test_set + set)) == 0:
            continue

        train_set = train_dataset(args.train_set + set)
        test_set = test_dataset(args.test_set + set)

        embedding_train = model.encode(train_set, convert_to_tensor=True)
        embedding_test = model.encode(test_set, convert_to_tensor=True)

        cosine_scores = util.pytorch_cos_sim(embedding_test, embedding_train)
        _, indices = torch.topk(cosine_scores, dim=-1, k=3)
        if 'test' in args.output_dir:
            indices = indices[:, 0:2].to('cpu')
        else:
            indices = indices[:, 1:3].to('cpu')

        for idx, sentence in zip(indices, test_set):
            results.append({
                'qid': sentence['qid'],
                'query': sentence['query'],
                'decoration1': train_set[idx[0]],
                'decoration2': train_set[idx[1]],
                'answer': sentence['answer'] if 'train' in args.test_set else '',
            })
        all_len += len(results)
        with open(args.output_dir + set, 'w+') as f:
            f.writelines(json.dumps(result) + '\n' for result in results)
        results = []
    print(all_len)
