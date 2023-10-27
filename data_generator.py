"""
This file generates batech dataset for training and predicting
input_texts and target_texts is the list format of source sequence and target
input_vocab_size and target_vocab_size is the number of long and label base-word vocabulary size，
max_input_len and max_target_len is the max length of jene sentence。


"""
import tensorflow as tf
import numpy as np
from keras.utils import to_categorical

max_sentence_length = 24 # length of sentence is 20, and one name, three sign tokens(<s>,</s>,<unk>)


def pad_dictionary(dictionary, target_length):
    """
    given a fixed dict length like 10000 for transfer
    :param dictionary:
    :param target_length:
    :return:
    """

    current_length = len(dictionary)

    if current_length >= target_length:
        return dictionary

    # generate the padded string list
    padding_strings = [str(i) for i in range(current_length, target_length)]

    # add the numercial index to the dictory
    for i, padding_string in enumerate(padding_strings):
        dictionary[padding_string] = len(dictionary)

    return dictionary

def read_corpus(file, dictionary):
    """
    this function is to read a corpus file in each line and count the lines and make vocabulary
    :param file:
    :param dictionary:
    :return:
    """
    dictionary['<s>'] = len(dictionary)  # start
    dictionary['</s>'] = len(dictionary)  # end
    dictionary['<unk>'] = len(dictionary)  # unknow tokens
    dictionary[' '] = len(dictionary)  # padding tokens

    with open(file, encoding='utf-8') as f:
        # Read and store every line
        lines_count = 0
        for line in f.readlines():
            lines_count = lines_count + 1
            for word in line.split(' '):
                if word not in dictionary:
                    dictionary[word.strip('\n')] = len(dictionary)
        dictionary = dict([(char, i) for i, char in enumerate(dictionary)])

    # dictionary = pad_dictionary(dictionary, 2000)
    return dictionary, lines_count


def append_line_texts(file,line_count):
    # Read the source data file and read the lines
    line_texts = []  # Input each line in the corpus
    with open(file, encoding='utf-8') as f:
        for l_i, line in enumerate(f):
            if len(line_texts) < line_count:
                line_texts.append(line)
    return line_texts

def split_to_tokens(sent, is_source, src_dictionary, tgt_dictionary, max_sentence_length):
    '''
    This function takes in a sentence (source or target)
    and preprocess the sentency with various steps (e.g. removing punctuation)
    add the '<s>' in the begining and add many end symbol '</s>'
    '''

    src_unk_count = 0
    tgt_unk_count = 0
    # global src_max_sent_length, tgt_max_sent_length

    sent = '<s>' + ' ' + sent  # Append <s> token's ID to the beggining of source sentence
    # Remove punctuation and new-line chars

    sent = sent.replace('\n', '')

    sent_toks = sent.split(' ')
    for _ in range(max_sentence_length - len(sent_toks) - 1):  # append </s> token's ID to the end
        sent_toks.append('<unk>')  # 其余weight填充空字符 the other weghts are padded with unk
    sent_toks.append('</s>')  # the last is end token to to stop the predict
    for t_i, tok in enumerate(sent_toks):
        if is_source:
            # src_dictionary contain the word -> word ID mapping for source vocabulary
            if tok not in src_dictionary.keys():
                if not len(tok.strip()) == 0:
                    sent_toks[t_i] = '<unk>'
                    src_unk_count += 1
        else:
            # tgt_dictionary contain the word -> word ID mapping for target vocabulary
            if tok not in tgt_dictionary.keys():
                if not len(tok.strip()) == 0:
                    sent_toks[t_i] = '<unk>'
                    # print(tok)
                    tgt_unk_count += 1
    return sent_toks


def split_to_tokens_infer(sent, src_dictionary, max_sentence_length):
    '''
    split the reads to be tokens
    '''

    src_unk_count = 0

    sent = '<s>' + ' ' + sent  # Append <s> token's ID to the beggining of source sentence

    # Remove punctuation and new-line chars
    sent = sent.replace(',', ' ,')
    sent = sent.replace('.', ' .')
    sent = sent.replace('\n', '')

    sent_toks = sent.split(' ')
    for _ in range(max_sentence_length - len(sent_toks) - 1):  # append <unk> in the blank space not </s>,it will decode stop early
        sent_toks.append('<unk>')
    sent_toks.append('</s>')# append </s> token's ID to the end
    for t_i, tok in enumerate(sent_toks):
        # src_dictionary contain the word -> word ID mapping for source vocabulary
        if tok not in src_dictionary.keys():
            if not len(tok.strip()) == 0:
                sent_toks[t_i] = '<unk>'
                src_unk_count += 1
    return sent_toks
# =====================================================================================

class Seq2SeqDataGenerator(tf.keras.utils.Sequence):
    """
    define to generate the datasets to train for seq2seq
    [encoder_input_data, decoder_input_data], decoder_target_data
    """

    def __init__(self, input_texts, target_texts, input_vocab_size, target_vocab_size, max_input_len, max_target_len,
                 batch_size, input_chars, target_chars):
        self.input_texts = input_texts  # input tokens sentences
        self.target_texts = target_texts
        self.input_vocab_size = input_vocab_size  # length of dictionary
        self.target_vocab_size = target_vocab_size
        self.max_input_len = max_input_len  # length of sentence
        self.max_target_len = max_target_len
        self.batch_size = batch_size
        self.input_chars = input_chars  # list of long reads-long_vocabulary
        self.target_chars = target_chars

        # build the map between gege word and numercial value
        self.input_token_index = dict([(char, i) for i, char in enumerate(input_chars)])  # input_token_index，词汇字典映射
        self.target_token_index = dict([(char, i) for i, char in enumerate(target_chars)])

    def __len__(self):
        return int(np.ceil(len(self.input_texts) / float(self.batch_size)))

    def __getitem__(self, idx):
        batch_input_texts = self.input_texts[idx * self.batch_size: (idx + 1) * self.batch_size]
        batch_target_texts = self.target_texts[idx * self.batch_size: (idx + 1) * self.batch_size]

        encoder_input_data = np.zeros((self.batch_size, self.max_input_len, self.input_vocab_size), dtype='float32')
        decoder_input_data = np.zeros((self.batch_size, self.max_target_len, self.target_vocab_size), dtype='float32')
        decoder_target_data = np.zeros((self.batch_size, self.max_target_len, self.target_vocab_size), dtype='float32')

        for i, (input_text, target_text) in enumerate(zip(batch_input_texts, batch_target_texts)):
            # reverse sentences in the encode layer of the seq2seq
            input_text_tokens = reversed(split_to_tokens(input_text, True, self.input_chars, self.target_chars, max_sentence_length))
            #input_text_tokens = split_to_tokens(input_text, True, self.input_chars, self.target_chars, max_sentence_length)
            target_text_tokens = split_to_tokens(target_text, False, self.input_chars, self.target_chars, max_sentence_length) # token label

            for t, char in enumerate(input_text_tokens):
                encoder_input_data[i, t, self.input_token_index[char]] = 1.
            encoder_input_data[i, t + 1:, self.input_token_index['</s>']] = 1.  # 为源句子的末尾加上终止字符，bubble是加空格
            for t, char in enumerate(target_text_tokens):
                decoder_input_data[i, t, self.target_token_index[char]] = 1.
                if t > 0:
                    decoder_target_data[i, t - 1, self.target_token_index[char]] = 1.
            decoder_input_data[i, t + 1:, self.target_token_index['</s>']] = 1.
            decoder_target_data[i, t:, self.target_token_index['</s>']] = 1.

        return [encoder_input_data, decoder_input_data], decoder_target_data



class predict_Seq2SeqDataGenerator(tf.keras.utils.Sequence):
    """
    define to generate the encoder_input_data dataset for prediction
    but in prediction, the target is predicted and concated in time,cant be catch previous
    return [encoder_input_data, decoder_input_data], None
    """

    def __init__(self, input_texts, target_texts, input_vocab_size, target_vocab_size, max_input_len, max_target_len,
                 batch_size, input_chars, target_chars):
        self.input_texts = input_texts  #
        self.target_texts = target_texts
        self.input_vocab_size = input_vocab_size  #
        self.target_vocab_size = target_vocab_size
        self.max_input_len = max_input_len  #
        self.max_target_len = max_target_len
        self.batch_size = batch_size
        self.input_chars = input_chars  #
        self.target_chars = target_chars


        self.input_token_index = dict([(char, i) for i, char in enumerate(input_chars)])  # input_token_index，词汇字典映射
        self.target_token_index = dict([(char, i) for i, char in enumerate(target_chars)])

    def __len__(self):
        return int(np.ceil(len(self.input_texts) / float(self.batch_size)))

    def __getitem__(self, idx):
        batch_input_texts = self.input_texts[idx * self.batch_size: (idx + 1) * self.batch_size]
        batch_target_texts = self.target_texts[idx * self.batch_size: (idx + 1) * self.batch_size]

        encoder_input_data = np.zeros((self.batch_size, self.max_input_len, self.input_vocab_size), dtype='float32')
        decoder_input_data = np.zeros((self.batch_size, self.max_target_len, self.target_vocab_size), dtype='float32')

        for i, (input_text, target_text) in enumerate(zip(batch_input_texts, batch_target_texts)):
            # token here
            input_text_tokens = reversed(split_to_tokens(input_text, True, self.input_chars, self.target_chars, max_sentence_length))
            #input_text_tokens = split_to_tokens(input_text, True, self.input_chars, self.target_chars, max_sentence_length)

            for t, char in enumerate(input_text_tokens):
                encoder_input_data[i, t, self.input_token_index[char]] = 1.
            encoder_input_data[i, t + 1:, self.input_token_index['</s>']] = 1.  

            # for t in range(len(input_text_tokens)):
            #     decoder_input_data[i, t, self.target_token_index['<s>']] = 1.  # 将每个句子的第一位写成起始字符

        # return [encoder_input_data, decoder_input_data], None
        return [encoder_input_data]

