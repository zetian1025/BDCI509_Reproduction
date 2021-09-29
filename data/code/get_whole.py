import os
import json
import argparse
import math


def read_inputs(input_file):
    with open(input_file) as f:
        lines = f.readlines()
        data = [json.loads(l.strip()) for l in lines]
    return data


def main():
    parser = argparse.ArgumentParser("divide.py")

    parser.add_argument("--data_type", type=str, default="dev")
    parser.add_argument("--data_dir", type=str,
                        default="./data/self_made/suffix2/")

    args = parser.parse_args()
    set_list = os.listdir(args.data_dir + args.data_type)
    whole = []

    for set in set_list:
        train_data = read_inputs(args.data_dir + args.data_type + '/' + set)
        whole += train_data

    with open(args.data_dir + args.data_type + '.txt', 'w+') as f:
        f.writelines(json.dumps(result) + '\n' for result in whole)


if __name__ == "__main__":
    main()
