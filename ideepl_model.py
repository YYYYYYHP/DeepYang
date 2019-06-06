# -*- coding: utf-8 -*-
"""
Created on Mon Apr 29 10:21:56 2019

@author: Dreamy
"""

from keras.layers import Input,Conv1D,MaxPooling1D,Bidirectional
from keras.layers.core import Activation, Dense, Dropout
from keras.layers.merge import concatenate
from keras.layers.recurrent import LSTM
from keras.models import Model
from keras import backend as K
K.set_image_dim_ordering('th')

class ideepl_model():
    def __init__(self):  
        print ("init")  
        
    @staticmethod
    def build_LSTM(seq_shape,structure_shape):
#    def build_LSTM(): 
        # input
        seq_input =  Input(shape = (111,4))
        structure_input = Input(shape = (111,6))
        
        #generate seq structure model
        seq_model = LSTM(64)(seq_input)
        seq_model = Dropout(0.3)(seq_model)
        
        structure_model = LSTM(64)(structure_input)
        structure_model = Dropout(0.3)(structure_model)
        
        #merge
        merged = concatenate([seq_model,structure_model],axis = -1)
        merged = Dense(128)(merged)
        merged = Activation("relu")(merged)
        merged = Dropout(0.4)(merged)
        
        merged = Dense(2)(merged)
        output = Activation("softmax")(merged)
        
        model = Model(inputs=[seq_input,structure_input],outputs= output)
        
        model.summary()
        
        return model
    
    @staticmethod
    def build_LSTM_CNN():
#    def build_LSTM(): 
        # input
        nbfilter = 16
        seq_input =  Input(shape = (111,4))
        structure_input = Input(shape = (111,6))
        
        
        #generate seq structure model
        seq_model = Conv1D(filters=nbfilter,kernel_size=10,input_shape=(111,4))(seq_input)
        seq_model_act = Activation("relu")(seq_model)
        seq_model_pool = MaxPooling1D(pool_size=3)(seq_model_act)
        seq_model_pool_D = Dropout(0.3)(seq_model_pool)
        seq_model_LSTM = LSTM(68)(seq_model_pool_D)
        seq_model_LSTM_D = Dropout(0.3)(seq_model_LSTM)
        
        structure_model = Conv1D(filters=nbfilter,kernel_size=10,input_shape=(111,4))(structure_input)
        structure_model_act = Activation("relu")(structure_model)
        structure_model_pool = MaxPooling1D(pool_size=3)(structure_model_act)
        structure_model_pool_D = Dropout(0.3)(structure_model_pool)
        structure_model_LSTM = LSTM(68)(structure_model_pool_D)
        structure_model_LSTM_D = Dropout(0.3)(structure_model_LSTM)
        
        #merge
        merged = concatenate([seq_model_LSTM_D,structure_model_LSTM_D],axis = -1)
        merged_Des = Dense(136)(merged)
        merged_act = Activation("relu")(merged_Des)
        merged_Dro = Dropout(0.4)(merged_act)
        
        merged_out = Dense(2)(merged_Dro)
        output = Activation("softmax")(merged_out)
        
        model = Model(inputs=[seq_input,structure_input],outputs= output)
        
        model.summary()
        
        return model
    
    @staticmethod
    def build_LSTM_CNN_BLSTM():
#    def build_LSTM(): 
        # input
        nbfilter = 16
        seq_input =  Input(shape = (111,4))
        structure_input = Input(shape = (111,6))
        
        
        #generate seq structure model
        seq_model = Conv1D(filters=nbfilter,kernel_size=10,input_shape=(111,4))(seq_input)
        seq_model_act = Activation("relu")(seq_model)
        seq_model_pool = MaxPooling1D(pool_size=3)(seq_model_act)
        seq_model_pool_D = Dropout(0.3)(seq_model_pool)
        seq_model_LSTM = LSTM(68,return_sequences=True)(seq_model_pool_D)
        seq_model_LSTM_D = Dropout(0.3)(seq_model_LSTM)
        
        structure_model = Conv1D(filters=nbfilter,kernel_size=10,input_shape=(111,4))(structure_input)
        structure_model_act = Activation("relu")(structure_model)
        structure_model_pool = MaxPooling1D(pool_size=3)(structure_model_act)
        structure_model_pool_D = Dropout(0.3)(structure_model_pool)
        structure_model_LSTM = LSTM(68,return_sequences=True)(structure_model_pool_D)
        structure_model_LSTM_D = Dropout(0.3)(structure_model_LSTM)
        

        #merge
        merged = concatenate([seq_model_LSTM_D,structure_model_LSTM_D],axis = -1)
        
        #BLSTM
        bilstm = Bidirectional(LSTM(2*nbfilter))(merged)
        bilstm = Dropout(0.3)(bilstm)
        
        
        bilstm = Dense(2)(bilstm)
        output = Activation("softmax")(bilstm)
        
        model = Model(inputs=[seq_input,structure_input],outputs= output)
        
        model.summary()
        
        return model
    
    @staticmethod
    def build_LSTM_B():
        ncell = 64
#    def build_LSTM(): 
        # input
        seq_input =  Input(shape = (111,4))
        structure_input = Input(shape = (111,6))
        
        #generate seq structure model
        seq_model = Bidirectional(LSTM(ncell))(seq_input)
        seq_model = Dropout(0.3)(seq_model)
        
        structure_model = Bidirectional(LSTM(ncell))(structure_input)
        structure_model = Dropout(0.3)(structure_model)
        
       
        #merge
        merged = concatenate([seq_model,structure_model],axis = -1)

        merged = Dense(2*ncell)(merged)
        merged = Activation("relu")(merged)
        merged = Dropout(0.4)(merged)
        
        merged = Dense(2)(merged)
        output = Activation("softmax")(merged)
        
        model = Model(inputs=[seq_input,structure_input],outputs= output)
        
        model.summary()
        
        return model
    
    @staticmethod
    def build_CNN_BLISTM(): 
        nbfilter =16
        # input
        seq_input =  Input(shape = (111,4))
        structure_input = Input(shape = (111,6))
        
        #generate seq structure model
        seq_model = Conv1D(filters=nbfilter,kernel_size=10,input_shape=(111,4))(seq_input)
        seq_model = Activation("relu")(seq_model)
        seq_model = MaxPooling1D(pool_size=3)(seq_model)
        seq_model = Dropout(0.2)(seq_model)
        
        structure_model = Conv1D(filters=nbfilter,kernel_size=10,input_shape=(111,6))(structure_input)
        structure_model = Activation("relu")(structure_model)
        structure_model = MaxPooling1D(pool_size=3)(structure_model)
        structure_model = Dropout(0.2)(structure_model)
        
        #merge
        merged = concatenate([seq_model,structure_model],axis = -1)
      
        #BLSTM
        bilstm = Bidirectional(LSTM(2*nbfilter))(merged)
        bilstm = Dropout(0.1)(bilstm)
        
        model = Dense(2*nbfilter)(bilstm)
        model = Activation("relu")(model)
        
        
        model = Dense(2)(model)
        output = Activation("softmax")(model)
        
        model = Model(inputs=[seq_input,structure_input],outputs= output)
        
        model.summary()
        return model

if __name__ == "__main__":
    ideepl_model.build()