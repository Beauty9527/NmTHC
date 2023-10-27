"""
this file is try to merge the trained and covered regions and uncovered regions
used once after each transfer
"""

import argparse

def build_parser():
    """Setup option parsing for sample."""
    parser = argparse.ArgumentParser(description='Split the mpileup into any parts you want.')
    parser.add_argument('--data-dir', type=str, help='The absolute path to long_corpus file.')
    parser.add_argument('--predict-fasta', type=str, help='The output of the predicted fasta.')
    parser.add_argument('--un-label-region', type=str, help='the uncovered regions.')
    parser.add_argument('--output-file', type=str, help='output-file for each trunk')

    args = parser.parse_args()

    return args

parsed_args = build_parser()
data_dir = parsed_args.data_dir
predict_fasta = data_dir +parsed_args.predict_fasta
un_label_region = data_dir +parsed_args.un_label_region
output_file = data_dir + parsed_args.output_file

def extract_sequence_info(line):
    parts = line.split(">")
    if len(parts) == 2:
        name, position_str = parts[1].split("&")
        return name.strip(), int(position_str)
    return None, None

def merge_sequences(file1, file2, output_file):
    temp_seqs = []
    sequences = []

    with open(file1, 'r') as f1:
        for line in f1:
            name, position = extract_sequence_info(line)
            if name is not None:
                sequence = next(f1).strip()
                temp_seqs.append((name, position, sequence))

    with open(file2, 'r') as f2:
        for line in f2:
            name, position = extract_sequence_info(line)
            if name is not None:
                sequence = next(f2).strip()
                temp_seqs.append((name, position, sequence))

    name_dict = []
    for seq in temp_seqs:
        if seq[0] not in name_dict:
            name_dict.append(seq[0])


    with open(output_file, 'w') as output:

        for i in range(len(name_dict)):
            current_name = name_dict[i]
            current_seq = ''
            temp_seq = []
            for seq in temp_seqs:
                if seq[0] == current_name:
                    temp_seq.append((seq[1], seq[2]))

            temp_seq.sort(key=lambda x: x[0])
            for item in temp_seq:
                current_seq += item[1]
            # sequences.append((current_name, current_seq))
            #print(current_name,":" '\t',len(current_seq))
            output.write(f">{current_name}\n{current_seq}\n")


merge_sequences(predict_fasta, un_label_region, output_file)
