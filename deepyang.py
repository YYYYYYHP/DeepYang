# -*- coding: utf-8 -*-
"""
Created on Mon Apr 29 09:33:47 2019

@author: Dreamy
"""
import os
import argparse
import gzip
import numpy as np
import pdb
import random
import matplotlib.pyplot as plt

from deepyang_model import *

import tensorflow as tf

from keras import backend as K
from keras.utils import np_utils
from keras.callbacks import TensorBoard
from keras.models import load_model

from sklearn.preprocessing import LabelEncoder
from sklearn.metrics import roc_auc_score

from rnashape_structure import run_rnashape
 

PATH = "C:/Users/Dreamy/.spyder-py3/workspace/class/iDeepL/"

#PATH = "/home/iDeepL/"


def get_RNA_seq_concolutional_array(seq, motif_len = 10):
    seq = seq.replace('U', 'T')  
    alpha = 'ACGT'
    #for seq in seqs:
    #for key, seq in seqs.iteritems():
    half_len = int(motif_len/2)
    row = int(len(seq) + half_len *2 )
    new_array = np.zeros((row, 4))    #(111 ,4)
    for i in range(half_len):
        new_array[i] = np.array([0.25]*4)   #前5行为0.25
    
    for i in range(row-half_len, row):     #后5行为0.25
        new_array[i] = np.array([0.25]*4)
        
    #pdb.set_trace()
    for i, val in enumerate(seq):  #枚举
        i = i + motif_len-1     #9-109
        if val not in 'ACGT':
            new_array[i] = np.array([0.25]*4)
            continue
        #if val == 'N' or i < motif_len or i > len(seq) - motif_len:
        #    new_array[i] = np.array([0.25]*4)
        #else:
        try:
            index = alpha.index(val)
            new_array[i][index] = 1
        except:
            pdb.set_trace()
        #data[key] = new_array
    return new_array


def get_RNA_structure_concolutional_array(seq, fw, structure = None, motif_len = 10):
    if fw is None:
        struc_en = structure
    else:
        #print 'running rnashapes'
        seq = seq.replace('T', 'U')
        struc_en = run_rnashape(seq)
        fw.write(struc_en.encode(encoding="utf-8") + b'\n')
        
    alpha = 'FTIHMS'
    half_len = int(motif_len/2)
    row = int(len(struc_en) +  half_len* 2)
    new_array = np.zeros((row, 6))
    for i in range(half_len):
        new_array[i] = np.array([0.16]*6)
    
    for i in range(row-half_len, row):
        new_array[i] = np.array([0.16]*6)

    for i, val in enumerate(struc_en):
        i = i + motif_len-1
        if val not in alpha:
            new_array[i] = np.array([0.16]*6)
            continue
        try:
            index = alpha.index(val)
            new_array[i][index] = 1
        except:
            pdb.set_trace()
        
    return new_array, struc_en

def read_rnashape(structure_file):
    struct_dict = {}
    with gzip.open(structure_file, 'rt') as fp:
        print(fp)
        for line in fp:
            if line[0] == '>':
                name = line[:-1]
            else:
                strucure = line[:-1]
                struct_dict[name] = strucure
    return struct_dict


def read_structure(seq_file, path):
    seq_list = []
    structure_list = []
    struct_exist = False
    if not os.path.exists(path + '/structure.gz'):
        fw = gzip.open(path + '/structure.gz', 'w')
    else:
        fw = None
        struct_exist = True
        struct_dict = read_rnashape(path + '/structure.gz')
        #pdb.set_trace()
    seq = ''
    old_name=''
    with gzip.open(seq_file, 'rt') as fp:
        for line in fp:
            if line[0] == '>':
                name = line
                if len(seq):
                    if struct_exist:
                        structure = struct_dict[old_name[:-1]]
                        seq_array, struct = get_RNA_structure_concolutional_array(seq, fw, structure = structure)
                    else:
                        fw.write(old_name.encode(encoding="utf-8"))
                        seq_array, struct = get_RNA_structure_concolutional_array(seq, fw)
                    seq_list.append(seq_array)
                    structure_list.append(struct)
                old_name = name
                
                seq = ''
            else:
                seq = seq + str(line[:-1])
        if len(seq): 
            if struct_exist:
                structure = struct_dict[old_name[:-1]]
                seq_array, struct = get_RNA_structure_concolutional_array(seq, fw, structure = structure)
            else:
                fw.write(old_name.encode(encoding="utf-8"))
                seq_array, struct = get_RNA_structure_concolutional_array(seq, fw)
            #seq_array, struct = get_RNA_structure_concolutional_array(seq, fw)
            seq_list.append(seq_array)
            structure_list.append(struct)  
    if fw:
        fw.close()
    return np.array(seq_list), structure_list


def read_seq(seq_file):
    seq_list = []
    label_list = []
    seq = ''
    with gzip.open(seq_file, 'rt') as fp:
        for line in fp:
            if line[0] == '>':
                name = line[1:-1]
                posi_label = name.split(';')[-1]             
                label = posi_label.split(':')[-1]  #class :0 or 1
                label_list.append(int(label))
                if len(seq):
                    seq_array = get_RNA_seq_concolutional_array(seq)
                    seq_list.append(seq_array)                    
                seq = ''
            else:
                seq = seq + str(line[:-1])
        if len(seq):
            seq_array = get_RNA_seq_concolutional_array(seq)  #(111,4)  9-109  1   0-4 0.25    106-110  1 or 0.25
            seq_list.append(seq_array) 
    
    return np.array(seq_list),np.array(label_list)

def split_training_validation(classes, validation_size = 0.2, shuffle = False):    #shuffle 洗牌
    """split sampels based on balnace classes"""
    num_samples=len(classes)
    classes=np.array(classes)
    classes_unique=np.unique(classes)
#    num_classes=len(classes_unique)
    indices=np.arange(num_samples)
    #indices_folds=np.zeros([num_samples],dtype=int)
    training_indice = []
    training_label = []
    validation_indice = []
    validation_label = []
    for cl in classes_unique:
        indices_cl=indices[classes==cl]
        num_samples_cl=len(indices_cl)

        # split this class into k parts
        if shuffle:
            random.shuffle(indices_cl) # in-place shuffle
        
        # module and residual
        num_samples_each_split=int(num_samples_cl*validation_size)
        res=num_samples_cl - num_samples_each_split
        
        training_indice = training_indice + [val for val in indices_cl[num_samples_each_split:]]
        training_label = training_label + [cl] * res
        
        validation_indice = validation_indice + [val for val in indices_cl[:num_samples_each_split]]
        validation_label = validation_label + [cl]*num_samples_each_split

    training_index = np.arange(len(training_label))
    random.shuffle(training_index)
    training_indice = np.array(training_indice)[training_index]
    training_label = np.array(training_label)[training_index]
    
    validation_index = np.arange(len(validation_label))
    random.shuffle(validation_index)
    validation_indice = np.array(validation_indice)[validation_index]
    validation_label = np.array(validation_label)[validation_index]    
    
            
    return training_indice, training_label, validation_indice, validation_label        
        


def load_data_file(inputfile, seq = True, onlytest = False):
    """
        Load data matrices from the specified folder.
    """
    path = os.path.dirname(inputfile)
    data = dict()
    if seq: 
        tmp = []
        seq_array,laebl_array =  read_seq(str(inputfile))
        tmp.append(seq_array)
        seq_onehot, structure = read_structure(inputfile, path)
        tmp.append(seq_onehot)
        data["seq"] = tmp
        data["structure"] = structure

    if onlytest:
        data["Y"] = []
    else:
        data["Y"] = laebl_array
        
    return data

def preprocess_labels(labels, encoder=None, categorical=True):
    if not encoder:
        encoder = LabelEncoder()
        encoder.fit(labels)
    y = encoder.transform(labels).astype(np.int32)
    if categorical:
        y = np_utils.to_categorical(y,2)
    return y, encoder

def array_change(array_data):
    data_array = np.zeros((len(array_data),array_data[0].shape[0],array_data[0].shape[1]),dtype = np.float)
    for i in range(len(array_data)):
        for j in range(array_data[i].shape[0]):
            for k in range(array_data[i].shape[1]):
                data_array[i,j,k] = array_data[i][j][k]
    return data_array    
        
def plot_acc_loss(history):
    # plot accuracy and loss plot
    plt.subplot(211)
    plt.title("Accuracy")
    plt.plot(history.history["acc"], color="g", label="train")
    plt.plot(history.history["val_acc"], color="b", label="validation")
    plt.legend(loc="best")
    
    plt.subplot(212)
    plt.title("Loss")
    plt.plot(history.history["loss"], color="g", label="train")
    plt.plot(history.history["val_loss"], color="b", label="validation")
    plt.legend(loc="best")
    
    plt.tight_layout()
    plt.show()
    
def train_ideepl(data_file, train_op, batch_size, nb_epoch,onlyLoad=False,get_AUC=True):
    name =  data_file.split("/")[3] 
    print("download data ")
    test_file = data_file.split("/")
    test_file[-2] = "test_sample_0" 
    test_file = "/".join(test_file)

    training_data = load_data_file(data_file)
    train_Y = training_data["Y"]
    training_indice, training_label, validation_indice, validation_label = split_training_validation(train_Y)  # 数据拆分   8:2
    print("get training data ")
    y, encoder = preprocess_labels(training_label)                                #按类别编码
    val_y, encoder = preprocess_labels(validation_label, encoder = encoder)
    
    test_data = load_data_file(test_file)
    test_Y = preprocess_labels(test_data["Y"])[0]
    print("get test data ")
    
    
    seq_data = training_data["seq"][0]
    seq_data = array_change(seq_data)
    seq_train = seq_data[training_indice.astype('int64')]
    seq_validation = seq_data[validation_indice.astype('int64')] 
    
    struct_data = training_data["seq"][1]
    struct_data = array_change(struct_data)
    struct_train = struct_data[training_indice.astype('int64')]
    struct_validation = struct_data[validation_indice.astype('int64')] 
    
    print("get training seq and struct data")

    seq_test = test_data["seq"][0]
    seq_test= array_change(seq_test)
    
    struct_test = test_data["seq"][1]
    struct_test= array_change(struct_test)
    
    print("get test seq and struct data ")

    if train_op == "LSTM_CNN":
        model  = ideepl_model.build_LSTM_CNN()
    elif train_op == "CNN_BLISTM":
        model = ideepl_model.build_CNN_BLISTM()
    elif train_op == "LSTM_B":
        model = ideepl_model.build_LSTM_B()
    elif train_op == "LSTM_CNN_BLSTM":
         model  = ideepl_model.build_LSTM_CNN_BLSTM()
    
    print("chose the "+train_op+" model")
    model_dir = "./"+train_op+"_models"
    if not os.path.exists(model_dir):
        os.makedirs(model_dir)

    if not onlyLoad:
    
        print("model built")
        model.compile(optimizer = "rmsprop",loss = "categorical_crossentropy",metrics=["accuracy"])
        print("model compiled")
        
        history = model.fit([seq_train,struct_train],[y],
                            batch_size=batch_size,epochs=nb_epoch,
                            validation_data=([seq_validation,struct_validation],[val_y]),
                            verbose=1,callbacks=[TensorBoard(log_dir="./"+train_op+"_logs/"+name)])

#        plot_acc_loss(history)
    
        if not get_AUC:
            score = model.evaluate([seq_test,struct_test],test_Y,verbose=1)
            print("Test score :",score[0])  #loss
            print("Test accuracy :",score[1])
        else :    
            fw = open(train_op+"_auc.txt","a+")
            pre = model.predict([seq_test,struct_test])
            auc = roc_auc_score(test_Y, pre)
            fw.write(name+":"+str(auc)+"\n")
            print ("Test AUC: ", auc)
            
        save_path = os.path.join(model_dir,name+'.pkl')
        model.save(save_path)
        print("model saved path:",save_path)


    
def predict_ideepl(data_file,train_op, out_file, onlytest = True):
    name =  data_file.split("/")[3] 
    
    test_data = load_data_file(data_file, onlytest= onlytest)

    print ('predicting')
    seq_test = test_data["seq"][0]
    seq_test= array_change(seq_test)
    
    struct_test = test_data["seq"][1]
    struct_test= array_change(struct_test)
    print(seq_test[0])
    print(struct_test[0])
    print("get test seq and struct data ")
    
    model_dir = "./"+train_op+"_models"
    
    model = load_model(os.path.join(model_dir,name+'.pkl')) 
    
    pred = model.predict([seq_test,struct_test])
    if not os.path.exists('./predicts/'):
        os.makedirs("./predicts/")
    fw = open("./predicts/"+out_file, 'w')
    myprob = "\n".join(map(str, pred[:, 1]))
    fw.write(myprob)
    fw.close()

def predict_ideepl_seq(seq, out_file):
    seq_array = [get_RNA_seq_concolutional_array(seq)]
    seq_ = seq.replace('T', 'U')
    print("seq:"+seq)
    print("seq_:"+seq_)
    struc_en = run_rnashape(seq_)
    struc_en = get_RNA_structure_concolutional_array(seq,structure=struc_en,fw=None)
    struct_arry = [struc_en[0]]
    if not os.path.exists(PATH+'seq_predicts/'):
        os.makedirs(PATH+"seq_predicts/")
    fw = open(PATH+"seq_predicts/"+out_file, 'w')
    fw.write("sequence:"+seq+"\n")
    fw.write("structure:"+struc_en[1]+"\n")
    fw.close()            
    print("get test seq and struct data ")

    file_path = PATH+"models"
    for model_name in os.listdir(file_path):
        model = load_model(os.path.join(file_path,model_name)) 
        pred = model.predict([seq_array,struct_arry])
        fw = open(PATH+"seq_predicts/"+out_file, 'a')
        myprob = "\n".join(map(str, pred[:, 1]))
        print(myprob)
        fw.write(model_name+":"+myprob+"\n")
        fw.close()



def run_ideepl(parser):
    data_file = parser.data_file
    train = parser.train
    train_op = parser.train_op
    predict = parser.predict
    predict_seq = parser.predict_seq
    sequence = parser.sequence
    batch_size = parser.batch_size
    n_epochs = parser.n_epochs
    train_all = parser.train_all
    
    data_file = "./datasets/clip/"+data_file+"/30000/training_sample_0/sequences.fa.gz"
    
 
    if train:
        print ('model training begin')
        train_ideepl(data_file,train_op, batch_size= batch_size, nb_epoch = n_epochs)
        print ('model training end')
    if predict:
        print ('model prediction begin')
        predict_ideepl(data_file, out_file = out_file, onlytest = True)
        print ('model prediction end')
    if predict_seq:
        print ('model prediction of seq begin')
        predict_ideepl_seq(sequence, out_file = out_file)
        print ('model prediction of seq end')
    if train_all:
        print ('All model training begin')
        file_path = PATH+"datasets/clip/"
        for rna_name in os.listdir(file_path):
            data_file = "./datasets/clip/"+rna_name+"/30000/training_sample_0/sequences.fa.gz"
            print(data_file)
        print ('model training end')

    

def parse_arguments(parser):
    parser.add_argument('--data_file', type=str, metavar='<data_file>', help='the sequence file used for training, it contains sequences and label (0, 1) in each head of sequence.')
    parser.add_argument('--train', type=bool, default=False, help='use this option for training model')
    parser.add_argument('--train_op', type=str, default=False, help='use this option for choosing the methods of training model')
    parser.add_argument('--train_all', type=bool, default=False, help='use this option for training model of all datasets')
    parser.add_argument('--predict', type=bool, default=False,  help='Predicting the RNA-protein binding sites for your input sequences, if using train, then it will be False')
    parser.add_argument('--sequence', type=str, help='input the sequence of predicting')
    parser.add_argument('--predict_seq', type=bool, default=False,  help='Predicting the RNA-protein binding sites for your input sequence')
    parser.add_argument('--batch_size', type=int, default=50, help='The size of a single mini-batch (default value: 50)')
    parser.add_argument('--n_epochs', type=int, default=30, help='The number of training epochs (default value: 30)')
    args = parser.parse_args()
    return args


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    args = parse_arguments(parser)
    run_ideepl(args)