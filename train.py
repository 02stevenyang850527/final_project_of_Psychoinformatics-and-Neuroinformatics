import numpy as np
import pandas as pd
import pickle
from keras.models import Sequential, load_model
from keras.preprocessing.text import Tokenizer
from keras.preprocessing import sequence
from keras.layers import Dense, Embedding, Dropout
from keras.layers.normalization import BatchNormalization
from keras.layers import Conv1D, MaxPooling1D
from keras.layers.wrappers import Bidirectional
from keras.layers import LSTM,SimpleRNN,GRU
import keras.backend as K
from keras.layers.advanced_activations import LeakyReLU
from keras.optimizers import SGD, Adam
from keras.callbacks import CSVLogger, EarlyStopping, ModelCheckpoint
from keras import regularizers
from keras.constraints import maxnorm

embedding_dim = 200
split_ratio = 0.1 # the ratio of validation data
batch = 64
epoch_num = 400
train_path = 'train.csv'
#test_path = 'test.csv'

def get_embedding_dict(path):
    embedding_dict = {}
    with open(path,'r') as f:
        for line in f:
            values = line.split(' ')
            word = values[0]
            coefs = np.asarray(values[1:],dtype='float32')
            embedding_dict[word] = coefs
    return embedding_dict


def get_embedding_matrix(word_index,embedding_dict,num_words,embedding_dim):
    embedding_matrix = np.zeros((num_words,embedding_dim))
    for word, i in word_index.items():
        if i < num_words:
            embedding_vector = embedding_dict.get(word)
            if embedding_vector is not None:
                embedding_matrix[i] = embedding_vector
    return embedding_matrix


def read_data(path,train):
    data = pd.read_csv(path, header=None)
    label=[]
    if (train):
        print ('Parse the training data')
        txt = data.iloc[:,1].values
        label = data.iloc[:,0].values
    else:
        print ('Parse the testing data')
        txt = data.iloc[:,1].values

    return (txt,label) 


def split_data(X,Y,split_ratio):
    np.random.seed(5)
    indices = np.random.permutation(X.shape[0])
    
    X_data = X[indices]
    Y_data = Y[indices]
    
    num_validation_sample = int(split_ratio * X_data.shape[0] )
    
    X_train = X_data[num_validation_sample:]
    Y_train = Y_data[num_validation_sample:]

    X_val = X_data[:num_validation_sample]
    Y_val = Y_data[:num_validation_sample]

    return (X_train,Y_train),(X_val,Y_val)


def main():
    (txt_train,label) = read_data(train_path,train=True)
    #(txt_test,_) = read_data(test_path,train=False)
    label = label.reshape(-1,1)
    
    ######### prepocess
    print ('Convert to index sequences.')
    #corpus = txt_train + txt_test
    corpus = txt_train

    tokenizer = Tokenizer()
    tokenizer.fit_on_texts(corpus)
    word_index = tokenizer.word_index
    train_sequences = tokenizer.texts_to_sequences(txt_train)

    pickle.dump(tokenizer,open('tokenizer.pkl','wb'))
    print('Pad sequences:')
    X = sequence.pad_sequences(train_sequences)
    Y = np.concatenate((1-label,label), axis=1)
    print('Split data into training data and validation data')
    (x_train,y_train),(x_val,y_val) = split_data(X,Y,split_ratio)
    max_article_length = x_train.shape[1]

    #########
    print ('maxlen', max_article_length)
    print ('Get embedding dict from glove.')
    embedding_dict = get_embedding_dict('../NTUEE/ML2017/hw5/glove.6B/glove.6B.%dd.txt'%embedding_dim)
    print ('Found %s word vectors.' % len(embedding_dict))
    max_features = len(word_index) + 1  # i.e. the number of words
    print ('Create embedding matrix.')
    embedding_matrix = get_embedding_matrix(word_index,embedding_dict,max_features,embedding_dim)

    print('Build model...')

    csv_logger = CSVLogger('training_report.csv',append=True)
    earlystopping = EarlyStopping(monitor='val_acc', patience = 7, verbose=1, mode='max')
    checkpoint = ModelCheckpoint(filepath='best.h5',
                                 verbose=1,
                                 save_best_only=True,
                                 monitor='val_acc',
                                 mode='max')
    model = Sequential()
    model.add(Embedding(max_features,
                        embedding_dim,
                        weights=[embedding_matrix],
                        input_length=max_article_length,
                        trainable=False
                       )
            )
    model.add(GRU(128, dropout=0.2, recurrent_dropout=0.2, activation='tanh'))
    model.add(Dense(2, activation='sigmoid'))
    model.compile(loss='categorical_crossentropy', optimizer='adam', metrics=['acc'])
    model.summary()
    
    model.fit(x_train,y_train,epochs=epoch_num, batch_size=batch, validation_data=(x_val,y_val),callbacks=[earlystopping,checkpoint])

if __name__=='__main__':
    main()
