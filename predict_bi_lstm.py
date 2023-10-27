"""
version 4.0
predict the results with the saved model in per batch
write it out
stop condition add                 if (sampled_char == '</s>' or word_num > max_target_len):
"""
from tensorflow.keras.models import load_model

import numpy as np
import argparse
import data_generator
import define_models
import os
import tensorflow as tf

def check_file(file):
    if os.path.exists(file):
        print(file, " is exists!")
        os.remove(file)

max_input_len = 24
max_target_len = 24
batch_size = 32

latent_dim = 256

def build_parser():
    """Setup option parsing for sample."""
    parser = argparse.ArgumentParser(description='Open the long_corpus  to generate the infer dataset in batch and predict the model.')
    parser.add_argument('--gpu', type=int, help='The device number you want.')
    parser.add_argument('--data-dir', type=str, help='The absolute path to long_corpus file.')
    parser.add_argument('--long-corpus', type=str, help='The name of long_corpus file.')
    parser.add_argument('--label-corpus', type=str, help='The name of label_corpus file.')
    parser.add_argument('--check-point-path', type=str, help='Path to save checkpoint file.')
    parser.add_argument('--model-hdf5-path', type=str, help='Path to save model.')
    parser.add_argument('--output', type=str, help='The predicted .fasta file.')
    args = parser.parse_args()

    return args


def batch_decode_sequence(generator, num_batches):
    all_decoded_sentence_batch = []
    for batch_index in range(num_batches):

        input_seq_batch = generator[batch_index][0]

        encoder_outputs, state_h, state_c = encoder_model.predict(input_seq_batch)

        target_seq_batch = np.zeros((len(input_seq_batch), 1, target_vocab_size))

        target_seq_batch[:, 0, label_vocabulary['<s>']] = 1.

        stop_condition = False
        decoded_sentence_batch = [''] * len(input_seq_batch)
        word_num = 0
        while not stop_condition:
            output_tokens_batch, h, c = decoder_model.predict(
                [target_seq_batch] + [state_h, state_c])


            sampled_token_index_batch = np.argmax(output_tokens_batch[:, -1, :], axis=-1)
            word_num += 1
            for i in range(len(input_seq_batch)):
                sampled_token_index = sampled_token_index_batch[i]
                sampled_char = label_reverse_vocabulary[sampled_token_index]
                decoded_sentence_batch[i] += sampled_char


                # if (sampled_char == '</s>'):
                #     stop_condition = True
                if (sampled_char == '</s>' or word_num > max_target_len):
                    stop_condition = True


                target_seq_batch = np.zeros((len(input_seq_batch), 1, target_vocab_size))
                for i in range(len(input_seq_batch)):
                    target_seq_batch[i, 0, sampled_token_index_batch[i]] = 1.

                state_h, state_c = [h, c]

        all_decoded_sentence_batch += decoded_sentence_batch

    return all_decoded_sentence_batch


def find_region(seq_name):

    input_string = seq_name

    ampersand_index = input_string.find('&')

    if ampersand_index != -1 and ampersand_index < len(input_string) - 1:
        digit_start_index = ampersand_index + 1
        while digit_start_index < len(input_string) and input_string[digit_start_index].isdigit():
            digit_start_index += 1

        non_digit_index = digit_start_index

        if non_digit_index < len(input_string):
            non_digit_character = input_string[non_digit_index]
            #print(f"The first non-digit character after '&' is '{non_digit_character}' at index {non_digit_index}.")
        else:
            print("No non-digit character found after '&'.")

    else:
        print("No valid '&' found in the input string.")

    return non_digit_index


def write_batch_seq(output_file, all_decoded_sentence_batch):
    with open(output_file, mode='a') as f:
        for line in all_decoded_sentence_batch[0:len(input_texts)]:
            if line.startswith('>') == True or line.startswith(' ') == True:
                start = find_region(line)

                seq_name = line[0: start]
                seq_value = line[start:].rstrip('</s>').strip('<unk>').replace('*','')
                f.write('\n')
                f.write(seq_name + '\n')
                f.write(seq_value)
            else:
                seq_value = line.rstrip('</s>').strip('<unk>').replace('*','')
                f.write(seq_value)

    f.close()


parsed_args = build_parser()
gpu_index = parsed_args.gpu
data_dir = parsed_args.data_dir
long_corpus = data_dir + parsed_args.long_corpus
label_corpus = data_dir + parsed_args.label_corpus
check_point_path = data_dir + parsed_args.check_point_path
model_hdf5_path = data_dir + parsed_args.model_hdf5_path
output_file = data_dir + parsed_args.output
check_file(output_file)

long_vocabulary = dict()
# Read the long_corpus file
long_vocabulary, long_lines_count = data_generator.read_corpus(long_corpus, long_vocabulary)
input_vocab_size = len(long_vocabulary)
# Build a reverse dictionary with the mapping ID -> word string
input_reverse_vocabulary = dict(zip(long_vocabulary.values(),long_vocabulary.keys()))
print("input_vocab_size:\t",input_vocab_size)

# Building labels vocabulary
# Contains label string -> ID mapping
label_vocabulary = dict()
# Read the long_corpus file
label_vocabulary, label_lines_count = data_generator.read_corpus(label_corpus, label_vocabulary)
target_vocab_size = len(label_vocabulary)
# Build a reverse dictionary with the mapping ID -> word string

label_reverse_vocabulary = dict(zip(label_vocabulary.values(),label_vocabulary.keys()))
print("target_vocab_size:\t", target_vocab_size)

input_texts = data_generator.append_line_texts(long_corpus, long_lines_count)
target_texts = data_generator.append_line_texts(label_corpus, label_lines_count)
# Make sure we extracted same number of both extracted source and target sentences


gpus = tf.config.experimental.list_physical_devices('GPU')

if gpus:
    tf.config.experimental.set_visible_devices(gpus[gpu_index], 'GPU')

predict_generator = data_generator.predict_Seq2SeqDataGenerator(input_texts, target_texts
                                                , input_vocab_size, target_vocab_size
                                                , max_input_len, max_target_len, batch_size
                                                , long_vocabulary, label_vocabulary
                                                )

model = load_model(model_hdf5_path)

_, encoder_model, decoder_model = define_models.bi_lstm_model(input_vocab_size, target_vocab_size, latent_dim)

encoder_model.load_weights(model_hdf5_path, by_name=True)
#encoder_model.summary()
decoder_model.load_weights(model_hdf5_path, by_name=True)
#decoder_model.summary()

all_decoded_sentence_batch = batch_decode_sequence(predict_generator, num_batches=len(input_texts)//batch_size + 1)  # len(input_texts)//batch_size+1


write_batch_seq(output_file, all_decoded_sentence_batch)

print("Predict success, the file is stored in:", output_file)

