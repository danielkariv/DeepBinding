import os
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from keras.models import Sequential
from keras.layers import Conv1D, MaxPooling1D, Flatten, Dense, Reshape
from keras.preprocessing.text import Tokenizer
from keras.preprocessing.sequence import pad_sequences
from keras.utils import to_categorical
from keras.callbacks import TensorBoard

decoder_weights_path = 'decoder_weights.h5'
classifier_weights_path = 'classifier_weights.h5'

def createAutoEncoderModel():
    ''' Create an AutoEncoder Model'''
    # TODO: we should try to extract features, based on chatgpt - using k-mer frequency, sequence motifs and such.. to capture characteristics related to binding. 
    # TODO: The model is okay.. (accuracy: ~0.83, loss: ~0.29) but I think could be much improved.
    length,digits = 41,6
    input_size = (length,digits)
    l1,l2 = 64,32
    # Define the encoder architecture
    encoder = Sequential(name='Encoder')
    encoder.add(Reshape((length*digits,), input_shape=input_size))  # Flatten the input to (120,)
    encoder.add(Dense(l1, activation='relu'))
    encoder.add(Dense(l2, activation='relu'))
    # Define the decoder architecture
    decoder = Sequential(name='Decoder')
    decoder.add(Dense(l1, activation='relu', input_shape=(l2,)))
    decoder.add(Dense(length * digits, activation='hard_sigmoid')) # binary output.
    decoder.add(Reshape(input_size))  # Reshape back to (20, 6)
    # Combine the encoder and decoder to create the autoencoder
    autoencoder = Sequential([encoder, decoder])
    autoencoder.summary()

    return autoencoder

def trainAutoEncoder(model, paths):
    ''' Trains the AutoEncoder based on files given, and the model.'''
    # TODO: maybe smarter way to process the RBNS files? Right now, it just runs over them, one by one, but maybe we should strength effect of training with higher density.
    # NOTE: Takes around 3min per epochs, so (5epochs*3min)*6files = 1.5 hours training on a M1 cpu.
    # hyperparms:
    epochs = 5
    print('processing RBNS data for RBP1')
    for path in paths:
        print('load data',path)
        x_train = pd.read_csv(path, delimiter='\t', header=None, usecols=[0])
        x_train = x_train[0].to_list()
        print('tokenizer')
        tokenizer = Tokenizer(char_level=True, num_words=6)
        tokenizer.fit_on_texts(x_train)
        sequences = tokenizer.texts_to_sequences(x_train)
        print('padding')
        #max_length = max(len(seq) for seq in sequences)
        max_length = 41
        padded_sequences = pad_sequences(sequences, maxlen=max_length)
        print('split train-test')
        x_train_seq, x_test_seq = train_test_split(padded_sequences, test_size=0.34, random_state=42)
        print('convert to one hot')
        x_train = to_categorical(x_train_seq)
        x_test = to_categorical(x_test_seq)
        
        # clear up
        sequences = None
        padded_sequences = None
        x_train_seq, x_test_seq = None, None

        print('start training')
        model.fit(x_train, x_train, epochs=epochs, batch_size=32, shuffle=True, validation_data=[x_test, x_test], callbacks=[TensorBoard(log_dir='/tmp/autoencoder')])

        # clear up
        x_train,x_test = None, None
    
    print('done processing, save weights for decoder')
    model.layers[0].save_weights(decoder_weights_path)

def createClassifierModel():
    ''' Creates a Classifier Model, it uses the encoding model from AutoEncoder and then runs on it's output to create binding scores.'''
    # TODO: it is really bad currently -> (0.00 accuracy, 2.29 loss). Need to play we the model design here.
    length,digits = 41,6
    input_size = (length,digits)
    l1,l2 = 64,32
    # Define the encoder architecture
    encoder = Sequential(name='Encoder')
    encoder.add(Reshape((length*digits,), input_shape=input_size))  # Flatten the input to (120,)
    encoder.add(Dense(l1, activation='relu'))
    encoder.add(Dense(l2, activation='relu'))
    # Define the decoder architecture
    classifier = Sequential(name='Classifier')
    classifier.add(Dense(l1, activation='relu', input_shape=(l2,)))
    classifier.add(Dense(l1, activation='relu'))
    classifier.add(Dense(1, activation='linear')) # single output (float)
    # Combine the encoder and decoder to create the autoencoder
    model = Sequential([encoder, classifier])
    model.summary()

    return model

def trainClassifierModel(model, path_seqs, path_rbp):
    ''' Trains the Classifier model based on sequences and their binidng values.'''
    # TODO: probably should train it over multiple RBPs to create something to use on general testing RBPs.
    # hyperparms:
    epochs = 5
    print('processing RNAcompete data for RBP1')
    print('load data',path_seqs)
    x_train = pd.read_csv(path_seqs, delimiter='\t', header=None, usecols=[0])
    x_train = x_train[0].to_list()
    print('tokenizer')
    tokenizer = Tokenizer(char_level=True, num_words=6)
    tokenizer.fit_on_texts('A C G U N')
    sequences = tokenizer.texts_to_sequences(x_train)
    print('padding')
    max_length = max(len(seq) for seq in sequences)
    padded_sequences = pad_sequences(sequences, maxlen=max_length)
    print('split train-test')
    x_train_seq, x_test_seq = train_test_split(padded_sequences, test_size=0.34, random_state=42)
    print('convert to one hot')
    x_train = to_categorical(x_train_seq)
    x_test = to_categorical(x_test_seq)
    print('processing RNCMPT for RBP1')
    y_train = pd.read_csv(path_rbp, delimiter='\t', header=None, usecols=[0])
    y_train = y_train[0].to_list()
    y_train, y_test = train_test_split(padded_sequences, test_size=0.34, random_state=42)
    # clear up
    sequences = None
    padded_sequences = None
    x_train_seq, x_test_seq = None, None

    print('start training')
    model.fit(x_train, y_train, epochs=epochs, batch_size=32, shuffle=True, validation_data=[x_test, y_test], callbacks=[TensorBoard(log_dir='/tmp/autoencoder')])

    # clear up
    x_train,x_test = None, None
    y_train,y_test = None, None
    print('done processing, save weights for classifier')
    model.layers[1].save_weights(classifier_weights_path)

def predictClassifierModel(model, path_seqs):
    ''' Predicts binding (output) for given sequences. '''
    # hyperparms:
    epochs = 5
    print('processing RNAcompete data for RBP1')
    print('load data',path_seqs)
    x_train = pd.read_csv(path_seqs, delimiter='\t', header=None, usecols=[0])
    x_train = x_train[0].to_list()
    print('tokenizer')
    tokenizer = Tokenizer(char_level=True, num_words=6)
    tokenizer.fit_on_texts('A C G U N')
    sequences = tokenizer.texts_to_sequences(x_train)
    print('padding')
    max_length = max(len(seq) for seq in sequences)
    padded_sequences = pad_sequences(sequences, maxlen=max_length)
    print('convert to one hot')
    seqs = to_categorical(padded_sequences)
    print('processing RNCMPT for RBP1')
    # clear up
    sequences = None
    padded_sequences = None
    print('start predicting')
    predictions = model_cls.predict(seqs,32)
    # save to file.
    np.savetxt('predictions.txt', predictions)

if __name__ == '__main__':
    # TODO: should recieve files paths from args, or atleast run training on known places but recieve in args the predictions part.
    # TODO: There is some repeated code in some places, like model creation, loading data and such.. ideally we should avoid it by reuse code with functions.
    # Create main model to predict outputs.
    model_cls = createClassifierModel()
    model_cls.compile(optimizer='adam', loss='mean_squared_error', metrics=['accuracy'])
    # Check if we got already RBP encoder weights ready, if so, we load them to main model, if not, we create them by training an AutoEncoder model.
    if os.path.exists(decoder_weights_path):
        model_cls.layers[0].load_weights(decoder_weights_path)
    else:
        print('missing encoder for RBP1. create model')
        model = createAutoEncoderModel()
        print('compile model')
        model.compile(optimizer='adam', loss='mean_squared_error', metrics=['accuracy'])
        trainAutoEncoder(model,['RBNS_training/RBP1_input.seq', 'RBNS_training/RBP1_5nM.seq','RBNS_training/RBP1_20nM.seq','RBNS_training/RBP1_80nM.seq','RBNS_training/RBP1_320nM.seq','RBNS_training/RBP1_1300nM.seq'])
        model_cls.layers[0] = model.layers[0] # Suppose to load weights in the next model. didn't test if right.
    # Check if we got already RBP scoring weights ready, if so, we load them to main model, if not we create them by training the model over data.
    
    if os.path.exists(classifier_weights_path):
        model_cls.layers[1].load_weights(classifier_weights_path)
    else:
        # TODO: It is wrong to train on known outputs for the one we try to predict, we should do them over some chunk of known RBPs, and test on the knowns others.
        #       It just takes too long to process, so I didn't care yet to implement it.
        trainClassifierModel(model_cls,'RNAcompete_sequences.txt','RNCMPT_training/RBP1.txt')
    # Run predictions and save it in a file.
    predictClassifierModel(model_cls,'RNAcompete_sequences.txt')
    
    


    

   
    