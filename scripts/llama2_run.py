import argparse
import logging

from llama2 import inference


def main():
    logging.basicConfig(level=logging.INFO, format='%(asctime)s:%(name)s:%(levelname)s:%(message)s')
    parser = argparse.ArgumentParser(description='Running a LLM model')
    parser.add_argument('--seed', type=int, default=None, help='Seed for debugging')
    parser.add_argument('--tokenizer-file', type=argparse.FileType('rb'), default="./data/tokenizer.bin", help='Tokenizer binary file')
    parser.add_argument('model_file', type=argparse.FileType('rb'), default="./data/stories15M.bin", nargs='?', help='Model binary file')
    parser.add_argument('temperature', type=float, default=0.0, nargs='?', help='Temperature')
    parser.add_argument('steps', type=int, default=256, nargs='?', help='Number of steps')
    parser.add_argument('prompt', type=str, default=None, nargs='?', help='Initial prompt')
    args = parser.parse_args()
    inference.run(args.model_file, args.tokenizer_file, args.temperature, args.steps, args.prompt, args.seed)


if __name__ == "__main__":
    main()
