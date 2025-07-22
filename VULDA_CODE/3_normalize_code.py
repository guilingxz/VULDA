# coding=utf-8
import os
import re
import argparse
from clean_gadget import clean_gadget

def parse_options():
    parser = argparse.ArgumentParser(description='Normalize source code files.')
    parser.add_argument('-i', '--input', help='Path to input directory containing .c files', type=str, required=True)
    return parser.parse_args()

def normalize_directory(dir_path):
    """
    Recursively normalizes all .c files in the input directory.
    It removes comments and applies clean_gadget preprocessing.
    """
    for root, _, files in os.walk(dir_path):
        for filename in files:
            if filename.endswith(".c"):
                filepath = os.path.join(root, filename)
                print(f"Normalizing: {filepath}")
                normalize_file(filepath)

def normalize_file(filepath):
    """
    Normalize a single .c file:
    - Remove comments (// and /* */)
    - Apply custom clean_gadget function
    """
    # Remove comments
    with open(filepath, "r") as file:
        code = file.read()
    code = re.sub(r'(?<!:)\/\/.*|\/\*(\s|.)*?\*\/', "", code, flags=re.DOTALL)
    with open(filepath, "w") as file:
        file.write(code.strip())

    # Apply clean_gadget logic
    with open(filepath, "r") as file:
        original_lines = file.readlines()
        normalized_lines = clean_gadget(original_lines)

    with open(filepath, "w") as file:
        file.writelines(normalized_lines)

def main():
    args = parse_options()
    normalize_directory(args.input)

if __name__ == '__main__':
    main()
