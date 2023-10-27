

import numpy as np
import pandas as pd
import os
import argparse
import re

label_symbols = ["*", "A", "C", "G", "T"]
feature_symbols = ["a", "c", "g", "t", "A", "C", "G", "T", "*", "#"]
word_length = 101
sentence_length = 20



def find_insertions(base_pileup):
    """Finds all of the insertions in a given base's pileup string.

    Args:
        base_pileup: Single base's pileup string output from samtools mpileup
    Returns:
        insertions: list of all insertions in pileup string
        next_to_del: whether insertion is next to deletion symbol (should be ignored)
    """
    insertions = []
    idx = 0
    next_to_del = []
    while idx < len(base_pileup):
        if base_pileup[idx] == "+" and base_pileup[idx+1].isdigit():
            end_of_number = False
            insertion_bases_start_idx = idx+1
            while not end_of_number:
                if base_pileup[insertion_bases_start_idx].isdigit():
                    insertion_bases_start_idx += 1
                else:
                    end_of_number = True
            insertion_length = int(base_pileup[idx:insertion_bases_start_idx])
            inserted_bases = base_pileup[insertion_bases_start_idx:insertion_bases_start_idx+insertion_length]
            insertions.append(inserted_bases)
            next_to_del.append(True if base_pileup[idx - 1] in '*#' else False)
            idx = insertion_bases_start_idx + insertion_length + 1  # skip the consecutive base after insertion
        else:
            idx += 1
    return insertions, next_to_del


def calculate_positions(start_pos, end_pos, subreads, sr_coverage):
    """Calculates positions array from read pileup columns.

    Args:
        start_pos: Starting index of pileup columnsyjf
        end_pos: Ending index of pileup columns
        subreads: Array of subread strings from pileup file
        exclude_no_coverage_positions: Boolean specifying whether to include 0 coverage
        positions during position calculation
    Returns:
        positions: Array of tuples containing major/minor positions of pileup
    """
    positions = []
    # Calculate ref and insert positions

    for i in range(start_pos, end_pos):

        if sr_coverage[i] == 0:
            lr_pos = []
            lr_pos.append([i, 0, 0]) #i is the original pos,0 is the num inserted, the last is the sr_coverage
            positions += lr_pos
            continue
        else:
            if type(subreads[i]) == float:
                subreads[i] = str('*')
            base_pileup = subreads[i].strip("^]").strip("$")

            # Get all insertions in pileup
            insertions, next_to_del = find_insertions(base_pileup)

            # Find length of maximum insertion
            longest_insertion = len(max(insertions, key=len)) if insertions else 0

            # Keep track of ref and insert positions in the pileup and the insertions
            # in the pileup.
            ref_insert_pos = []  # ref position for ref base pos in pileup, insert for additional inserted bases
            ref_insert_pos.append([i, 0, 1])
            for j in range(longest_insertion):
                ref_insert_pos.append([i, j + 1, 1])  # the last 1 means there is a SR coverage
            positions += ref_insert_pos

    return positions

def reencode_base_pileup(ref_base, pileup_str, del_current):
    """Re-encodes mpileup output of list of special characters to list of nucleotides.

    Args:
        ref_base : Reference nucleotide
        pileup_str : mpileup encoding of pileup bases relative to reference
    Returns:
        A pileup string of special characters replaced with nucleotides.
        del_current:如果置1则为当前位置发生了删除，相应解码为*
    """
    pileup = []
    if del_current == 0:
        for c in pileup_str:
            if c == "." or c == "*":
                pileup.append(ref_base)
            elif c == ",":
                pileup.append(ref_base.lower())
            else:
                pileup.append(c)
    else:
        pileup.append("*")

    return "".join(pileup)


def long_absolute(pileup):
    """
    According to the alignment reencode the long reads in absolute position
    :param pileup:
    :return:
    """
    start_pos = 0
    end_pos = len(pileup)
    subreads = pileup[:, 4]
    sr_coverage = pileup[:,3].astype("int")
    positions = calculate_positions(start_pos, end_pos, subreads, sr_coverage)
    positions = np.array(positions)

    lr_bases = ''

    for i in range(len(positions)):
        ref_position = positions[i][0]
        insert_position = positions[i][1]
        if insert_position == 0:
            lr_bases += pileup[ref_position, 2]
        elif insert_position > 0:
            lr_bases += '*'

    return lr_bases,positions

def short_absolute(pileup):
    """
    generate labels string for each seq in LR set with the information in SR reads which aligned to it.
    :param pileup:
    :return:
    """
    start_pos = 0
    end_pos = len(pileup)
    subreads = pileup[:, 4]
    sr_coverage = pileup[:, 3].astype("int")
    positions = calculate_positions(start_pos, end_pos, subreads, sr_coverage)
    positions = np.array(positions)
    labels = ''

    num_features = len(feature_symbols)

    pileup_counts = np.zeros(shape=(len(positions), num_features + 1), dtype=np.float32)
    for i in range(len(positions)):
        ref_position = positions[i][0]
        insert_position = positions[i][1]
        if type(subreads[ref_position]) == float:
            subreads[i] = str('*')
        base_pileup = subreads[ref_position].strip("^]").strip("$")
        temp = re.search(r"(?<=-)\d+", base_pileup)
        if temp != None:
            deletion_num = re.search(r"(?<=-)\d+", base_pileup).group(0)
            for del_num in range(int(deletion_num)):
                pileup_counts[i + del_num + 1, 9] = 1
            keep_chars = ",.*"
            result = ''.join([c for c in base_pileup if c in keep_chars])
            base_pileup = reencode_base_pileup(pileup[ref_position, 2], result,
                                               pileup_counts[i, 9])
            insertions, next_to_del = find_insertions(base_pileup)
            insertions_to_keep = []
            if sr_coverage[ref_position] != 0.:
                pileup_counts[i, 10] = 1

            # Remove all insertions which are next to delete positions in pileup
            for k in range(len(insertions)):
                if next_to_del[k] is False:
                    insertions_to_keep.append(insertions[k])

            # Replace all occurrences of insertions from the pileup string
            for insertion in insertions:
                base_pileup = base_pileup.replace("+" + str(len(insertion)) + insertion, "")

            if insert_position == 0:  # No insertions for this position
                for j in range(len(feature_symbols) - 1):
                    pileup_counts[i, j] = base_pileup.count(feature_symbols[j])
                # Add draft base and base quality to encoding

            elif insert_position > 0:
                # Remove all insertions which are smaller than minor position being considered
                # so we only count inserted bases at positions longer than the minor position
                insertions_minor = [x for x in insertions_to_keep if len(x) >= insert_position]
                for j in range(len(insertions_minor)):
                    inserted_base = insertions_minor[j][insert_position - 1]
                    if "ATGCatgc".find(inserted_base) == -1:
                        inserted_base = '*'
                    pileup_counts[i, feature_symbols.index(inserted_base)] += 1

            if not (np.any(pileup_counts[i, 0:9])):
                label_base = "N"
            else:
                label_base = feature_symbols[np.argmax(pileup_counts[i, 0:9])].upper()

        else:
            base_pileup = reencode_base_pileup(pileup[ref_position, 2], base_pileup, pileup_counts[i, 9])  #  改了reencode函数，使feature从长读本身来
            insertions, next_to_del = find_insertions(base_pileup)
            insertions_to_keep = []
            if sr_coverage[ref_position] != 0.:
                pileup_counts[i, 10] = 1

            # Remove all insertions which are next to delete positions in pileup
            for k in range(len(insertions)):
                if next_to_del[k] is False:
                    insertions_to_keep.append(insertions[k])

            # Replace all occurrences of insertions from the pileup string
            for insertion in insertions:
                base_pileup = base_pileup.replace("+" + str(len(insertion)) + insertion, "")

            if insert_position == 0:  # No insertions for this position
                for j in range(len(feature_symbols) - 1):
                    pileup_counts[i, j] = base_pileup.count(feature_symbols[j])
                # Add draft base and base quality to encoding

            elif insert_position > 0:
                # Remove all insertions which are smaller than minor position being considered
                # so we only count inserted bases at positions longer than the minor position
                insertions_minor = [x for x in insertions_to_keep if len(x) >= insert_position]
                for j in range(len(insertions_minor)):
                    inserted_base = insertions_minor[j][insert_position - 1]
                    if "ATGCatgc".find(inserted_base) == -1:
                        inserted_base = '*'
                    pileup_counts[i, feature_symbols.index(inserted_base)] += 1

            if not(np.any(pileup_counts[i, 0:9])):
                label_base = "N"
            else:
                label_base = feature_symbols[np.argmax(pileup_counts[i, 0:9])].upper()


        labels = labels + label_base
    return labels



def write_clong(id, read, file):
    #print("id:",id,"\t", "length", len(read))
    write_clong.call_count += 1
    #print(f"This is write_long call #{write_clong.call_count}")
    with open(file, mode='a') as f:
        seq_name = ">" + str(id)
        region_order = "&" + str(write_clong.call_count) + "\n"
        f.write(seq_name)
        f.write(region_order)
        f.write(read)
        f.write("\n")
    f.close()


def write_unlong(id, read, file):

    with open(file, mode='a') as f:
        seq_name = ">" + str(id) + '\n'

        f.write(seq_name)
        f.write(read)
        f.write("\n")
    f.close()

def corpus(fasta_file, corpus_file, word_length, sentence_length):
    """
    preprocessed reads to corpus with equal length
    make the long_read with absolute position to long_corpus
    make the long_label with absolute position to label_corpus
    :param fasta_file: the file of the input long_reads file with absolute position
    :param corpus_file: the file of the output long_reads corpus with sentences format
    :param word_length: the length of the word set as 9 first
    :param sentence_length: the length of the sentences in each line
    :return:
    """
    with open(fasta_file, mode='r') as reads:
        with open(corpus_file,mode='a') as corpus:  # each read come to add,clear st the first
            for read in reads.readlines():
                read = read.strip('\n')
                end = 0
                if read.startswith('>'):
                    word = read.strip('\n')
                    corpus.write(word + ' ')
                else:
                    i = 0
                    for start in range(0, len(read) - word_length + 1, word_length):
                        end = start + word_length
                        word = read[start: end].strip('\n')
                        corpus.write(word)
                        i = i + 1
                        if i % sentence_length != 0:
                            corpus.write(' ')
                        else:
                            corpus.write('\n')
                    if len(read) >= end:
                        start = end
                        last_word = read[start:]
                        corpus.write(last_word + '\n')

    reads.close()
    corpus.close()


def split_regions(cur_name, long_read, positions, cover_output, uncover_output):  #
    c_regions = ''
    un_regions = ''
    write_clong.call_count = 0  # used to record the order of the region in the cur long read 区域顺序
    for i in range(len(long_read) - 1):  #
        if i == len(long_read) -2: # last regions
            if len(c_regions) != 0:
                c_regions += long_read[i:i+2]
                write_clong(cur_name, c_regions, cover_output)
            else:
                un_regions += long_read[i:i+2]
                write_clong(cur_name, un_regions, uncover_output)
        else:
            if positions[i, 2] == 0 and positions[i, 2] == positions[i + 1, 2]:
                un_regions += long_read[i]
            elif positions[i, 2] == 0 and positions[i, 2] != positions[i + 1, 2]:
                un_regions += long_read[i]
                write_clong(cur_name, un_regions, uncover_output)
                un_regions = ''
            elif positions[i, 2] == 1 and positions[i, 2] == positions[i + 1, 2]:
                c_regions += long_read[i]
            elif positions[i, 2] == 1 and positions[i, 2] != positions[i + 1, 2]:
                c_regions += long_read[i]
                write_clong(cur_name, c_regions, cover_output)
                c_regions = ''



def lr_label_build(mpileup_file):
    """
    take the pileup as input
    :return:
    """
    pileup_total = pd.read_csv(mpileup_file, delimiter="\t", header=None, quoting=3).values
    seq_name = pileup_total[:, 0]  # the first column of pileup is the seq_name set
    sequence = pileup_total[:, 2]
    sr_cov = pileup_total[:, 3]

    cur_start = 0
    seq_num = 1

    for i in range(0, len(pileup_total) - 1):
        if i == len(pileup_total) - 2:  # if the last seq
            cur_end = i
            sr = sr_cov[cur_start: i+2]

            if not (np.any(sr)):
                read = ''.join([e for e in sequence[cur_start: cur_end + 1]])
                write_unlong(seq_name[i], read, un_whole_long)
                read = ''
            else:
                pileup = pileup_total[cur_start:cur_end + 1, :]  # split pileup into per seq
                cur_name = seq_name[cur_start]

                long_read, positions = long_absolute(pileup)  # region for per seq
                label = short_absolute(pileup)
                split_regions(cur_name, long_read, positions, c_long_regions, un_long_regions)
                split_regions(cur_name, label, positions, c_label_regions, un_label_regions)

                seq_num += 1
                #print("seq_num:\t", seq_num)
                assert len(long_read) == len(label)

                if len(long_read) == 0 or len(label) == 0:  # if the seq has no coverage from SR or truth,ignore it
                    cur_start = i + 1
                    seq_num += 1
                    continue

        if seq_name[i] == seq_name[i + 1]:
            continue
        else:
            sr = sr_cov[cur_start: i + 1]
            cur_end = i


        if not (np.any(sr)):  #
            read = ''.join([e for e in sequence[cur_start: cur_end + 1]])
            write_unlong(seq_name[i], read, un_whole_long)
            read = ''
        else:
            pileup = pileup_total[cur_start:cur_end + 1, :]  # split pileup into per seq
            cur_name = seq_name[cur_start]

            long_read, positions = long_absolute(pileup)  # region for per seq
            label = short_absolute(pileup)
            split_regions(cur_name, long_read, positions, c_long_regions, un_long_regions)
            split_regions(cur_name, label, positions, c_label_regions, un_label_regions)

            if len(long_read) == 0 or len(label) == 0:  # if the seq has no coverage from SR or truth,ignore it
                cur_start = i + 1
                seq_num += 1
                continue

        cur_start = cur_end + 1
        seq_num += 1
        #print("seq_num:\t", seq_num)

    print("position array are equal")

    return 0


def check_file(file):
    if os.path.exists(file):
        #print(file, " is exists!")
        os.remove(file)



def build_parser():
    """Setup option parsing for sample."""
    parser = argparse.ArgumentParser(description='Make the corpus for the mpileup file.')
    parser.add_argument('--data_dir', type=str, help='The path of datasets foder')
    parser.add_argument('--input_file', type=str, help='Path of input pileup file.')
    parser.add_argument('--uncovered_file', type=str, help='Path of the uncovered whole long read uncovered.fasta.')
    parser.add_argument('--covered_long_regions', type=str, help='Path of the filtered covered_long_region.fasta.')
    parser.add_argument('--covered_label_regions', type=str, help='Path of the filtered covered_label_region.fasta.')
    parser.add_argument('--uncovered_long_regions', type=str, help='Path of the filtered uncovered_long_region.fasta.')
    parser.add_argument('--uncovered_label_regions', type=str, help='Path of the filtered uncovered_label_region.fasta.')
    # uncovered_long_regions and uncovered_label_regions should be the same
    parser.add_argument('--long_corpus', type=str,
                        help='the outputfile of corpus of long reads with absolute position in .txt format.')
    parser.add_argument('--label_corpus', type=str,
                        help='the outputfile of corpus of labels with absolute position in .txt format.')


    args = parser.parse_args()

    return args


def un_reads(mpileup, un_whole_long):
    df = pd.read_csv(mpileup, delimiter="\t", header=None, quoting=3)


    grouped = df.groupby(0)[2].apply(''.join)

    
    fasta_file = un_whole_long
    with open(fasta_file, 'w') as f:
        for index, sequence in grouped.iteritems():
            print(index, ":" '\t', len(sequence))
            f.write(f'>{index}\n')
            f.write(f'{sequence}\n')

if __name__ == '__main__':

    parsed_args = build_parser()

    data_dir = parsed_args.data_dir
    input_file = str(data_dir) + str(parsed_args.input_file)
    un_whole_long = data_dir + parsed_args.uncovered_file
    c_long_regions = data_dir + parsed_args.covered_long_regions
    c_label_regions = data_dir + parsed_args.covered_label_regions
    un_long_regions = data_dir + parsed_args.uncovered_long_regions
    un_label_regions = data_dir + parsed_args.uncovered_label_regions

    long_corpus = data_dir + parsed_args.long_corpus
    label_corpus = data_dir + parsed_args.label_corpus


    check_file(c_label_regions)
    check_file(un_whole_long)  # clear the output file to avoid overlying
    check_file(c_long_regions)  # clear the output file to avoid overlying
    check_file(un_long_regions)
    check_file(un_label_regions)

    check_file(long_corpus)
    check_file(label_corpus)
    lr_label_build(input_file)


    if os.path.exists(c_long_regions):
        corpus(c_long_regions, long_corpus, word_length=word_length, sentence_length=sentence_length)
        corpus(c_label_regions, label_corpus, word_length=word_length, sentence_length=sentence_length)

    else:
        un_reads(input_file, un_whole_long)



