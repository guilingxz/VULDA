import os
import re
import hashlib
import argparse
from tqdm import tqdm  # tqdm progress bar

def extract_functions(source_code):
    """
    Extracts functions from source code text.

    Parameters:
    - source_code (str): The full source code as a string.

    Returns:
    - functions (list): A list of extracted function code blocks.
    """
    functions = []
    in_function = False
    function_lines = []
    bracket_count = 0

    for line in source_code.splitlines():
        if not in_function:
            match = re.match(r"(\w+\s*\**\s+\w+\s*\([^)]*\))", line)
            if match:
                in_function = True
                function_lines = [match.group(1)]
                bracket_count = 0

        if in_function:
            function_lines.append(line)
            for char in line:
                if char == '{':
                    bracket_count += 1
                elif char == '}':
                    bracket_count -= 1
                    if bracket_count == 0:
                        in_function = False
                        function_code = "\n".join(function_lines).strip()
                        # Skip one-line declarations with a semicolon unless more functions follow
                        if len(function_lines) == 1 and ';' in function_lines[0] and not re.search(
                                r"(\w+\s*\**\s+\w+\s*\([^)]*\))", source_code[source_code.index(function_code) + len(function_code):]):
                            break
                        functions.append(function_code)
                        break

    return functions


def main():
    parser = argparse.ArgumentParser(description="Extract functions from .c/.cpp source files in a batch.")
    parser.add_argument('-i', '--input', required=True, help='Input folder containing .c/.cpp files')
    parser.add_argument('-o', '--output', required=True, help='Output folder for extracted functions')
    args = parser.parse_args()

    input_folder = args.input
    output_folder = args.output

    os.makedirs(output_folder, exist_ok=True)

    # Get all .c and .cpp files from input folder
    source_files = [os.path.join(input_folder, f) for f in os.listdir(input_folder) if f.endswith((".c", ".cpp"))]

    for source_file in tqdm(source_files, desc="Processing files"):
        with open(source_file, 'r') as f:
            code = f.read()
            functions = extract_functions(code)

            for idx, function_code in enumerate(functions):
                # Generate MD5 hash as identifier
                function_code_hash = hashlib.md5(function_code.encode()).hexdigest()
                function_filename = f"{idx}_{function_code_hash}.c"
                target_file = os.path.join(output_folder, function_filename)

                try:
                    with open(target_file, 'w') as f_out:
                        # Skip the first line (repeated declaration)
                        f_out.write("\n".join(function_code.splitlines()[1:]) + "\n")
                except OSError as e:
                    print("Failed to write file:", function_filename)
                    print("Error:", str(e))


if __name__ == '__main__':
    main()
