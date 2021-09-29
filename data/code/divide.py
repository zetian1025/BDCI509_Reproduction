# 用于切割train集
import os
import json
import argparse
import math
from sklearn.model_selection import train_test_split

RATE = 0.7  # dev集占原始train集的比例


def read_inputs(input_file):
    with open(input_file) as f:
        lines = f.readlines()
        data = [json.loads(l.strip()) for l in lines]
    return data


def main():
    parser = argparse.ArgumentParser("divide.py")

    parser.add_argument("--train_dir", type=str,
                        default="./data/self_made/suffix2/train/")
    parser.add_argument("--dev_dir", type=str,
                        default="./data/self_made/suffix2/dev/")

    args = parser.parse_args()
    set_list = os.listdir(args.train_dir)

    for set in set_list:
        train_data = read_inputs(args.train_dir + set)

        new_train, dev = train_test_split(
            train_data, test_size=0.3, random_state=42)

        with open(args.train_dir + set, 'w') as f:
            f.writelines(json.dumps(result) + '\n' for result in new_train)

        with open(args.dev_dir + set, 'w') as f:
            f.writelines(json.dumps(result) + '\n' for result in dev)


if __name__ == "__main__":
    main()
