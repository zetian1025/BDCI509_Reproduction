import json
import os

import pandas as pd
from torch.utils.data.dataloader import DataLoader

import evaluate


import argparse
import random
import torch
import numpy as np
import torch.nn.utils as utils
from torch.optim import Adam
from tqdm import tqdm
from transformers import BertTokenizer, get_linear_schedule_with_warmup, AutoTokenizer, RobertaTokenizer
from model.modeling import BertForMultiMask
from data_utils.dataset import read_inputs, KCTDataset
from rules import mask_rule

from os.path import join


def seed_torch(seed=2021):
    random.seed(seed)
    os.environ['PYTHONHASHSEED'] = str(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.deterministic = True


def init_argparse():
    parser = argparse.ArgumentParser()
    parser.add_argument("--seed", type=int, default=2021, help="Random seed")
    parser.add_argument("--pretrained_model", type=str,
                        default="../bert-large-cased-whole-word-masking")

    parser.add_argument("--data_dir", type=str,
                        default="../data/self_made/suffix2")
    parser.add_argument("--train_file", type=str, default="train.txt")
    parser.add_argument("--dev_file", type=str, default="dev.txt")
    parser.add_argument("--test_file", type=str, default="test.txt")

    parser.add_argument("--gpu", type=str, default='0',
                        help='GPU ID to use [default: 1]')
    parser.add_argument("--save_path", type=str,
                        default="../save/bert-large-cased-whole-word-masking-prompt-1.pt")
    parser.add_argument("--template", type=str, default='(4, 3)')
    parser.add_argument("--rate", type=float, default=0.7)

    parser.add_argument("--write_dir", type=str, default="../output/")

    parser.add_argument("--batch_size", type=int, default=16)
    parser.add_argument("--lr", type=float, default=2e-6)

    parser.add_argument("--epoch", type=int, default=2)
    parser.add_argument("--decay_rate", type=float, default=0.98)

    parser.add_argument("--weight_decay", type=float, default=0)
    parser.add_argument("--num_warm_steps", type=int, default=200)

    parser.add_argument("--lstm_dropout", type=float, default=0.5)

    parser.add_argument("--pseudo_token", type=str, default='[unused2]')

    parser.add_argument("--only_generate", type=str, default='False')
    parser.add_argument("--data_type", type=str, default='dev')

    args = parser.parse_args()
    args.template = eval(args.template) if type(
        args.template) is not tuple else args.template
    print("the model is from {}".format(args.save_path))
    print("the rate is {}".format(args.rate))
    assert type(args.template) is tuple

    return args


class Trainer(object):
    def __init__(self, args):
        self.args = args
        self.device = 'cuda' if torch.cuda.is_available() else 'cpu'
        self.tokenizer = BertTokenizer.from_pretrained(args.pretrained_model)

        self.train_data = read_inputs(
            join(self.args.data_dir, args.train_file))
        self.dev_data = read_inputs(join(self.args.data_dir, args.dev_file))
        self.test_data = read_inputs(join(self.args.data_dir, args.test_file))

        self.train_set = KCTDataset('train', self.train_data)
        self.dev_set = KCTDataset('dev', self.dev_data)
        self.test_set = KCTDataset('test', self.test_data)

        self.train_loader = DataLoader(
            self.train_set, batch_size=self.args.batch_size, shuffle=True, drop_last=True, num_workers=8)
        self.dev_loader = DataLoader(
            self.dev_set, batch_size=self.args.batch_size, num_workers=8)

        self.model = BertForMultiMask(args, self.args.template, self.device)
        self.optimizer = Adam(self.model.parameters(
        ), lr=self.args.lr, weight_decay=self.args.weight_decay)
        self.train_steps = self.args.epoch * len(self.train_loader)
        self.schedule = get_linear_schedule_with_warmup(self.optimizer,
                                                        num_warmup_steps=self.args.num_warm_steps,
                                                        num_training_steps=self.train_steps)

    def do(self):
        if self.args.only_generate == 'True':
            self.generate()
        else:
            self.train()
            self.generate()

    def train(self):
        print("USE {} GPUs".format(torch.cuda.device_count()))

        best_epoch = -1
        best_loss = float('inf')

        for epoch in range(self.args.epoch):
            self.train_epoch(epoch)
            eval_loss = self.evaluating_epoch(epoch)
            print("\tBest={:.3f}, Current={:.3f}".format(best_loss, eval_loss))
            if eval_loss < best_loss:
                best_loss = eval_loss
                best_epoch = epoch
                torch.save(self.model.state_dict(), self.args.save_path)
                print("\tParam saved in epoch", epoch)
        print("Best Epoch:", best_epoch)

    def train_epoch(self, epoch):
        self.model.train()
        t_bar = tqdm(self.train_loader)
        for batch in t_bar:
            self.optimizer.zero_grad()
            loss, logits = self.model(batch[0], batch[1])

            loss.backward()

            utils.clip_grad_norm_(self.model.parameters(), 2.0)

            self.optimizer.step()
            self.schedule.step()
            t_bar.set_description(
                "Train Epoch{}  Loss:{:.5f}".format(epoch, loss))

    @torch.no_grad()
    def evaluating_epoch(self, epoch):
        self.model.eval()

        sum_loss, cnt = 0, 0
        d_bar = tqdm(self.dev_loader)
        for batch in d_bar:
            loss, _ = self.model(batch[0], batch[1])
            sum_loss += loss
            d_bar.set_description(
                "Dev Epoch{}  Loss:{:.5f}".format(epoch, loss))
            cnt += 1
        return sum_loss / cnt

    def generate(self):
        self.model.load_state_dict(torch.load(
            self.args.save_path, map_location='cuda:0'))
        self.model.eval()
        if self.args.data_type == 'dev':
            self.generator(self.dev_set)
            self.evaluate_dev()
        else:
            self.generator(self.test_set)

    def generator(self, data_set):
        results = []
        data_loader = DataLoader(data_set, batch_size=2, num_workers=8)
        bar = tqdm(data_loader)
        for batch in bar:

            _, right_text1 = batch[0][0].strip().split('[SEP]')
            _, right_text2 = batch[0][1].strip().split('[SEP]')
            assert right_text1 == right_text2

            top_k, rate = mask_rule.mask_count(right_text1)

            output = []

            for i, topk in enumerate(top_k):
                if topk == 0:
                    continue

                _, prior_logits = self.model(
                    [" [SEP] " + right_text1], ['[MASK] ' * (i + 1)])
                prior_logits1, prior_logits2 = prior_logits, prior_logits

                _, logits1 = self.model([batch[0][0]], ['[MASK] ' * (i + 1)])
                _, logits2 = self.model([batch[0][1]], ['[MASK] ' * (i + 1)])

                new_logits1 = logits1 - rate * prior_logits2
                new_logits2 = logits2 - rate * prior_logits2

                new_logits = args.rate*new_logits1 + (1-args.rate)*new_logits2

                new_logits, sorted_idx = torch.topk(new_logits, 10, dim=-1)

                predicted_tokens = []
                for k in range(topk):
                    predicted_token = self.tokenizer.decode(sorted_idx[:, k])
                    predicted_tokens.append(predicted_token)
                output.append(predicted_tokens)

            results.append({"id": batch[2][0],
                            "ret": json.dumps([rets for rets in [ans for element in output for ans in element]])})

        df = pd.DataFrame(results)
        df.to_csv(join(self.args.write_dir, '{}.csv'.format(
            self.args.data_type)), index=False)

    def evaluate_dev(self):
        result = evaluate.judge(join(self.args.data_dir, args.dev_file), join(
            self.args.write_dir, 'dev.csv'))
        print(result[0], end='')
        print(';', end='')
        print(result[1], end='')
        print(';', end='')
        print(result[2], end='')
        print('')


if __name__ == "__main__":
    args = init_argparse()
    seed_torch(args.seed)
    os.environ["CUDA_VISIBLE_DEVICES"] = args.gpu
    trainer = Trainer(args)
    trainer.do()
