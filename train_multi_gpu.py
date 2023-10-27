"""
version 3.0 双向lstm训练，时间减近似一半，数据输入结构不变，但是解码阶段各时间步输入状态有变
use keras to generate a seq2seq model using TF2.3
train the model by batch
the model used here is from define_models.py
the structure is more clear
"""

from tensorflow import keras
from tensorflow.keras.callbacks import EarlyStopping
import tensorflow as tf
import matplotlib.pyplot as plt
import argparse
import data_generator
import define_models
max_input_len = 24
max_target_len = 24
batch_size = 64
epochs = 150
# 256维神经元
latent_dim = 256

def build_parser():
    """Setup option parsing for sample."""
    parser = argparse.ArgumentParser(description='Open the long_corpus and the label_corpus to generate the train dataset in batch and train the model.')
    parser.add_argument('--data-dir', type=str, help='The absolute path to long_corpus file.')
    parser.add_argument('--long-corpus', type=str, help='The name of long_corpus file.')
    parser.add_argument('--label-corpus', type=str, help='The name of label_corpus file.')
    parser.add_argument('--check-point-path', type=str, help='Path to save checkpoint file.')
    parser.add_argument('--model-hdf5-path', type=str, help='Path to save model.')
    args = parser.parse_args()

    return args
# 解析运行参数
parsed_args = build_parser()
data_dir = parsed_args.data_dir
long_corpus = data_dir + parsed_args.long_corpus
label_corpus = data_dir + parsed_args.label_corpus
check_point_path = data_dir + parsed_args.check_point_path
model_hdf5_path = data_dir + parsed_args.model_hdf5_path


# Contains word string -> ID mapping
long_vocabulary = dict()
# Read the long_corpus file
long_vocabulary, long_lines_count = data_generator.read_corpus(long_corpus, long_vocabulary)
input_vocab_size = len(long_vocabulary)
# Build a reverse dictionary with the mapping ID -> word string
#input_reverse_vocabulary = dict(zip(long_vocabulary.values(),long_vocabulary.keys()))
input_token_index = dict([(char, i) for i, char in enumerate(long_vocabulary)])
print("input_vocab_size:\t",input_vocab_size)

# Contains label string -> ID mapping
label_vocabulary = dict()
# Read the long_corpus file
label_vocabulary, label_lines_count = data_generator.read_corpus(label_corpus, label_vocabulary)
target_vocab_size = len(label_vocabulary)
# Build a reverse dictionary with the mapping ID -> word string
# label_reverse_vocabulary = dict(zip(label_vocabulary.values(),label_vocabulary.keys()))
target_token_index = dict([(char, i) for i, char in enumerate(label_vocabulary)])
print("target_vocab_size:\t", target_vocab_size)

# Read the source data file and read the lines,and append them as a list for token

input_texts = data_generator.append_line_texts(long_corpus, long_lines_count)
target_texts = data_generator.append_line_texts(label_corpus, label_lines_count)
# Make sure we extracted same number of both extracted source and target sentences
assert len(input_texts) == len(target_texts), 'Source: %d, Target: %d' % (len(input_texts), len(target_texts))


gpus = tf.config.experimental.list_physical_devices('GPU')

if gpus:

    strategy = tf.distribute.MirroredStrategy(devices=[f"/gpu:{i}" for i in range(len(gpus))])

    with strategy.scope():

        model, encoder_model, decoder_model = define_models.bi_lstm_model(input_vocab_size, target_vocab_size,
                                                                          latent_dim)

        model.compile(optimizer='rmsprop', loss='categorical_crossentropy', metrics=['accuracy'])

        train_generator = data_generator.Seq2SeqDataGenerator(input_texts, target_texts
                                                              , input_vocab_size, target_vocab_size
                                                              , max_input_len, max_target_len, batch_size
                                                              , long_vocabulary, label_vocabulary
                                                              )
        callbacks = [keras.callbacks.ModelCheckpoint(filepath=check_point_path
                                                     , save_best_only=True
                                                     , save_weights_only=True
                                                     , monitor='loss'
                                                     , verbose=1, )
                     # ,keras.callbacks.EarlyStopping(monitor='loss', patience=5)
                     ]

        # early_stopping = EarlyStopping(monitor='loss', patience=10)

        history = model.fit(train_generator
                            # , steps_per_epoch=len(input_texts)//batch_size + 1  # 加了这句就提示数据不够是什么意思，早上训练还好好的？
                            , epochs=epochs
                            , callbacks=callbacks
                            , validation_data=train_generator
                            )
        model.save(model_hdf5_path)

        print("Training finished!")
        print("The weights and model are saved in:", check_point_path)


        plt.plot(history.history['accuracy'])
        plt.plot(history.history['loss'])
        plt.title('Model accuracy&loss')
        plt.xlabel('Epoch')
        plt.legend(['Train_acc', 'Train_loss'])
        # plt.show()

        plt.savefig('loss_curve.png')

        plt.close()


else:
    print("没有可用的GPU设备。")






