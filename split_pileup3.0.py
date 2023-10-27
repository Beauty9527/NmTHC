"""
split_pileup3.0 for the especial big data, with csv and a mode”a“
"""

import os
import argparse

def build_parser():
    """Setup option parsing for sample."""
    parser = argparse.ArgumentParser(description='Split the mpileup into any parts you want.')
    parser.add_argument('--data-dir', type=str, help='The absolute path to long_corpus file.')
    parser.add_argument('--whole-mpileup', type=str, help='The name of input mpileup to be split.')
    parser.add_argument('--split-rate', type=int, help='how many parts to divide it into')
    parser.add_argument('--output-dir', type=str, help='output-dir')

    args = parser.parse_args()

    return args

parsed_args = build_parser()
data_dir = parsed_args.data_dir
file_path = data_dir +parsed_args.whole_mpileup
num_chunks = parsed_args.split_rate
output_dir = data_dir + parsed_args.output_dir


for chunk_index in range(num_chunks):
    output_file_path = output_dir + f'chunk_{chunk_index}.mpileup'
    if os.path.exists(output_file_path):
        os.remove(output_file_path)


chunk_files = {}

with open(file_path, 'r') as file:
    for line in file:
        first_dimension_value = line.strip().split('\t')[0]

        chunk_index = hash(first_dimension_value) % num_chunks

        if chunk_index not in chunk_files:
            output_file_path = output_dir + f'chunk_{chunk_index}.mpileup'
            chunk_files[chunk_index] = open(output_file_path, 'a')  # 使用'a'模式以追加方式打开文件

        chunk_files[chunk_index].write(line)

for chunk_file in chunk_files.values():
    chunk_file.close()
