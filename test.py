import numpy as np
import pickle
from keras.models import load_model
from keras.preprocessing import sequence

batch = 64
test_path = 'test.csv'
output_path = 'ans.csv'

def read_data(path,train):
    data = open(path,'r').readlines()
    label=[]
    txt=[]
    if (train):
        print ('Parse the training data')
        for i in range(len(data)):
            d = data[i].find(',')
            temp = data[i][d+1:]
            label.append(int(data[i][:d]))
            txt.append(temp)
        label = np.array(label)
    else:
        print ('Parse the testing data')
        for i in range(len(data)):
            txt.append(data[i])

    return (txt,label) 


(txt_test,_) = read_data(test_path,train=False)
model = load_model('best.h5')
tokenizer = pickle.load(open('tokenizer.pkl','rb'))
word_index = tokenizer.word_index
test_sequences = tokenizer.texts_to_sequences(txt_test)

print('Pad sequences:')
x_test = sequence.pad_sequences(test_sequences,maxlen=36) # maxlen being known from train.py

print ('x_test shape:',x_test.shape)

result = model.predict(x_test, batch_size=batch, verbose=1)

of = open(output_path,'w')
out_txt = '"txt","prob"\n'

for i in range(len(result)):
    out_txt += txt_test[i] + ',' + str(result[i,0]) + '\n'

of. write(out_txt)
of.close()
