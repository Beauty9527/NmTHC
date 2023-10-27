"""
models define version 2.0
define 3 models, train needs 1,predict need 2 to updata states in prediction
"""

from tensorflow.keras.layers import Input, LSTM, Dense, GRU, Bidirectional, Embedding
from tensorflow.keras.models import Model
from tensorflow.python.keras.layers import Concatenate


def models(input_vocab_size, target_vocab_size, latent_dim):  # the size of input/output dictionary, latent_dim=number of neural
    # encoder
    encoder_inputs = Input(shape=(None, input_vocab_size))
    # LSTM
    encoder_lstm = LSTM(units=latent_dim, return_state=True)
    encoder_outputs, state_h, state_c = encoder_lstm(encoder_inputs)
    encoder_states = [state_h, state_c]

    # decoder
    decoder_inputs = Input(shape=(None, target_vocab_size))
    # decoder_gru = GRU(units=latent_dim, return_sequences=True, return_state=True)
    decoder_lstm = LSTM(units=latent_dim, return_sequences=True, return_state=True)
    decoder_outputs, _, _ = decoder_lstm(decoder_inputs, initial_state=encoder_states)

    # dense layer
    decoder_dense = Dense(units=target_vocab_size, activation='softmax')
    decoder_outputs = decoder_dense(decoder_outputs)

    # model
    model = Model([encoder_inputs, decoder_inputs], decoder_outputs)
    ##==============================================================
    # encoder
    encoder_model = Model(encoder_inputs, encoder_states)
    # decoder
    decoder_state_input_h = Input(shape=(latent_dim,))
    decoder_state_input_c = Input(shape=(latent_dim,))
    decoder_states_inputs = [decoder_state_input_h, decoder_state_input_c]
    decoder_outputs, state_h, state_c = decoder_lstm(decoder_inputs, initial_state=decoder_states_inputs)
    decoder_states = [state_h, state_c]
    decoder_outputs = decoder_dense(decoder_outputs)
    decoder_model = Model([decoder_inputs] + decoder_states_inputs, [decoder_outputs] + decoder_states)
    return model, encoder_model, decoder_model

def embedding_models(input_vocab_size, target_vocab_size, latent_dim):

    encoder_inputs = Input(shape=(None,))  # 定义模型输入维度
    encoder_embedding = Embedding(input_dim=input_vocab_size, output_dim=200)(encoder_inputs)  #

    encoder_lstm = LSTM(units=latent_dim, return_state=True)
    encoder_outputs, state_h, state_c = encoder_lstm(encoder_embedding)
    encoder_states = [state_h, state_c] #


    decoder_inputs = Input(shape=(None,))
    decoder_embedding = Embedding(input_dim= target_vocab_size, output_dim=200)(decoder_inputs)

    # decoder_gru = GRU(units=latent_dim, return_sequences=True, return_state=True)
    decoder_lstm = LSTM(units=latent_dim, return_sequences=True, return_state=True)
    decoder_outputs, _, _ = decoder_lstm(decoder_embedding, initial_state=encoder_states)

    #
    decoder_dense = Dense(units=target_vocab_size, activation='softmax')
    decoder_outputs = decoder_dense(decoder_outputs)

    #
    model = Model([encoder_inputs, decoder_inputs], decoder_outputs)
    ##==============================================================
    #
    encoder_model = Model(encoder_inputs, encoder_states)
    #
    decoder_state_input_h = Input(shape=(latent_dim,))
    decoder_state_input_c = Input(shape=(latent_dim,))
    decoder_states_inputs = [decoder_state_input_h, decoder_state_input_c]
    decoder_outputs, state_h, state_c = decoder_lstm(decoder_inputs, initial_state=decoder_states_inputs)
    decoder_states = [state_h, state_c]
    decoder_outputs = decoder_dense(decoder_outputs)
    decoder_model = Model([decoder_inputs] + decoder_states_inputs, [decoder_outputs] + decoder_states)
    #
    return model, encoder_model, decoder_model


def gru_models(input_dim, output_dim, hidden_units):
    """
    """
    #
    encoder_input = Input(shape=(None, input_dim))
    encoder_gru = GRU(hidden_units, return_state=True)
    encoder_outputs, state_h = encoder_gru(encoder_input)
    encoder_model = Model(encoder_input, state_h)

    #
    decoder_input = Input(shape=(None, output_dim))
    decoder_gru = GRU(hidden_units, return_sequences=True, return_state=True)
    decoder_outputs, _ = decoder_gru(decoder_input, initial_state=state_h)
    decoder_dense = Dense(output_dim, activation='softmax')
    decoder_outputs = decoder_dense(decoder_outputs)
    decoder_model = Model([decoder_input, state_h], [decoder_outputs])

    #
    model_input = [encoder_input, decoder_input]
    model_output = decoder_model([decoder_input, encoder_model(encoder_input)])
    model = Model(model_input, model_output)

    return model, encoder_model, decoder_model


def bi_lstm_model(input_vocab_size, target_vocab_size, latent_dim):


    encoder_inputs = Input(shape=(None, input_vocab_size))

    encoder_bi_lstm = Bidirectional(LSTM(latent_dim, return_sequences=True, return_state=True))

    encoder_outputs, forward_h, forward_c, backward_h, backward_c = encoder_bi_lstm(encoder_inputs)
    state_h = Concatenate()([forward_h, backward_h])
    state_c = Concatenate()([forward_c, backward_c])


    decoder_inputs = Input(shape=(None, target_vocab_size))

    decoder_lstm = LSTM(latent_dim * 2, return_sequences=True, return_state=True)

    decoder_outputs, _, _ = decoder_lstm(decoder_inputs, initial_state=[state_h, state_c])

    decoder_dense = Dense(target_vocab_size, activation='softmax')
    decoder_outputs = decoder_dense(decoder_outputs)


    model = Model([encoder_inputs, decoder_inputs], decoder_outputs)


    encoder_model = Model(encoder_inputs, [encoder_outputs, state_h, state_c])


    decoder_state_input_h = Input(shape=(latent_dim * 2,))
    decoder_state_input_c = Input(shape=(latent_dim * 2,))
    decoder_states_inputs = [decoder_state_input_h, decoder_state_input_c]


    decoder_outputs, state_h, state_c = decoder_lstm(decoder_inputs, initial_state=decoder_states_inputs)
    decoder_states = [state_h, state_c]
    decoder_outputs = decoder_dense(decoder_outputs)


    decoder_outputs, state_h, state_c = decoder_lstm(decoder_inputs, initial_state=decoder_states_inputs)
    decoder_states = [state_h, state_c]
    decoder_outputs = decoder_dense(decoder_outputs)

    
    decoder_model = Model([decoder_inputs] + decoder_states_inputs, [decoder_outputs] + decoder_states)
    return model, encoder_model, decoder_model
