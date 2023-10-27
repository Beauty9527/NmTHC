"""
remame the dumpliacted reads with tail
"""
import argparse
import os

def build_parser():
    """Setup option parsing for sample."""
    parser = argparse.ArgumentParser(description='Change the dumplicated name in the long read instead remove them.')
    parser.add_argument('--data_dir', type=str, help='The path of datasets foder')
    parser.add_argument('--input_file', type=str, help='Path of input fasta file.')
    parser.add_argument('--output_file', type=str, help='Path of output fasta file.')
    args = parser.parse_args()
    return args


if __name__ == '__main__':

    parsed_args = build_parser()

    data_dir = parsed_args.data_dir
    input_file = data_dir + parsed_args.input_file
    output_file = data_dir + parsed_args.output_file
    if os.path.exists(output_file):
        os.remove(output_file)

    name_list = []

    with open(input_file, 'r') as file:
        lines = file.readlines()


    with open(output_file, 'a') as output_file:

        modified_lines = []
        seq_num = 0
        for index, line in enumerate(lines):
            if index % 2 == 1:

                output_file.writelines(line)
            else: 
                seq_num += 1
                modified_lines = '>' + str(seq_num) + '\n'
                output_file.writelines(modified_lines)
    file.close()
    output_file.close()



