import subprocess
import os
import argparse
from tqdm import tqdm

def parse_source_code_to_dot(file_path, filename, joern_path, out_dir_cpg):
    """
    Parse a source code directory to CPG and export .dot format.

    Args:
        file_path (str): Path to the source code folder
        filename (str): Source code filename
        joern_path (str): Path to Joern CLI
        out_dir_cpg (str): Output directory for CPG .dot files
    """
    os.makedirs(out_dir_cpg, exist_ok=True)

    print(f'Parsing {filename} to CPG...')
    parse_cmd = f"sh {joern_path}/joern-parse {file_path}"
    subprocess.call(parse_cmd, shell=True)

    print(f'Exporting CPG to DOT...')
    export_cmd = f"sh {joern_path}/joern-export --repr cpg14 --out {os.path.join(out_dir_cpg, filename.split('.')[0])}"
    subprocess.call(export_cmd, shell=True)

def main():
    parser = argparse.ArgumentParser(description="Parse source code to Joern CPG DOT files.")
    parser.add_argument('-i', '--input', required=True, help='Root folder containing subfolders of .c/.cpp files')
    parser.add_argument('-o', '--output', required=True, help='Output folder to store exported DOT files')
    parser.add_argument('--joern-path', default='/home/se/joern-cli/', help='Path to Joern CLI (default: /home/se/joern-cli/)')
    args = parser.parse_args()

    input_dir = args.input
    output_dir = args.output
    joern_path = args.joern_path

    source_dirs = os.listdir(input_dir)
    for folder in tqdm(source_dirs, desc="Parsing source folders"):
        folder_path = os.path.join(input_dir, folder)
        output_cpg_path = os.path.join(output_dir, folder)

        # Skip already processed folders
        if os.path.exists(output_cpg_path) and len(os.listdir(output_cpg_path)) > 0:
            print(f'Skipping {folder_path}, already processed.')
            continue

        print(f'Starting to process {folder_path}...')
        try:
            source_file = os.listdir(folder_path)[0]  # Assumes one file per folder
            parse_source_code_to_dot(folder_path, source_file, joern_path, output_cpg_path)
        except IndexError:
            print(f'No source files found in {folder_path}, skipping.')

if __name__ == '__main__':
    main()
