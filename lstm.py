# -*- coding: utf-8 -*-

from keras.models import Sequential  
from keras.layers.core import Dense, Activation,Dropout
from keras.layers.recurrent import LSTM
from keras.regularizers import *

import pandas as pd
import numpy as np

from keras.utils import np_utils
from keras.callbacks import EarlyStopping


num4DF = pd.read_csv("data/num4.csv",names=("1","2","3","4"))

for i in xrange(1,5):
	num4DF["answer"+str(i)] = num4DF[str(i)].shift(-1)
num4DF.dropna(subset=["answer1"],inplace=True)


# 1~4桁目を質的変数とみなして変換
trainDF = pd.get_dummies(num4DF["1"])
train2Array = pd.get_dummies(num4DF["2"])
trainDF = pd.concat([trainDF, train2Array], axis=1)
train3Array = pd.get_dummies(num4DF["3"])
trainDF = pd.concat([trainDF, train3Array], axis=1)
train4Array = pd.get_dummies(num4DF["4"])
trainDF = pd.concat([trainDF, train4Array], axis=1)

# [0,1]に正規化
trainDF = trainDF.apply(lambda x: (x-x.min())/(x.max()-x.min()), axis=0).fillna(0)
training_data = np.array(trainDF.values.tolist())

# RNNの時はこっち
training_data = np.reshape(training_data, training_data.shape + (1,))

print training_data.shape[0]
print training_data.shape[1]

# 次の回に出てきた数を1桁ずつ答えにして学習させてみる
y_train1 = np.array(pd.get_dummies(num4DF["answer1"]).values.tolist())
y_train2 = np.array(pd.get_dummies(num4DF["answer2"]).values.tolist())
y_train3 = np.array(pd.get_dummies(num4DF["answer3"]).values.tolist())
y_train4 = np.array(pd.get_dummies(num4DF["answer4"]).values.tolist())

col_num = len(trainDF.columns)



def learningLSTM(answer,weightname):
	model = Sequential()
	model.add(LSTM(output_dim = 10,init='glorot_uniform', inner_init='orthogonal', forget_bias_init='one', activation='softmax',inner_activation='tanh', input_shape=training_data.shape[1:],return_sequences=False,W_regularizer=l1l2(l1=0.01, l2=0.01),dropout_U=0.1))
	model.compile(optimizer='RMSprop',
	              loss='categorical_crossentropy',
	              metrics=['accuracy'])
	print(model.summary())
	model.fit(training_data, answer, batch_size=600, nb_epoch=1000, validation_split=0.05)  
	model.save_weights("weight/LSTM"+weightname+'.hdf5')

learningLSTM(y_train1,"num1")
learningLSTM(y_train2,"num2")
learningLSTM(y_train3,"num3")
learningLSTM(y_train4,"num4")
