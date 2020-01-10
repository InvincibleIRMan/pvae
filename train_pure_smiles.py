from model_property2 import *
from data_prepration import *
from params import *
from multiprocessing import cpu_count
import numpy as np
import pickle
from util import *
import pickle 
import os
import pandas as pd
import torch.optim as optim
import torch.nn as nn
import json
Args = Params()

print('############### data loading ######################')
data_dir = 'data/'
data_list = ['train_zinc','valid_zinc','test_zinc']

data_train,data_valid,data_test = text2dict_zinc(data_dir, data_list)
#txt, data_train_ = smi_postprocessing(data_train,biology_flag=False,LB_smi=15,UB_smi=120)

print("Number of SMILES >>>> " , len(data_train['SMILES']))


# smiles preprocessing:
'''
We filter the SMILES based on two criterias:
1) 8 < logP < -3 and 600 < MolecularWeight < 50,
2) 120 < smiles with length <  20

 pre-processing step is happening in smi_postprocessing is function.
 
 Please note that, biology_flag is served for the situations that the user has an access to biological label.
####
'''
biology_flag = False
LogP = [-3,8]
MW = [50,600]
LB_smi= 20
UB_smi = 120
txt,data_train_ = smi_postprocessing(data_train,biology_flag,LB_smi,UB_smi,LogP,MW)


'''
Dictionary building part: 
In this part, we generate dictionary based on the output of the smi_postrocessing function.

Then, we shape the data to the usable format for the pytorch using dataset_building function,
followed by DataLoader function.

'''
data_type = 'pure_smiles'
char2ind,ind2char,sos_indx,eos_indx,pad_indx = dictionary_build(txt)
max_sequence_length = np.max(np.asarray([len(smi) for smi in data_train_['SMILES']]))
dataset_train = dataset_building(char2ind,data_train_,max_sequence_length,data_type)
dataloader_train = DataLoader(dataset=dataset_train, batch_size= Args.batch_size, shuffle = True)


'''
Model loading step: 

We defined the model with pvae fuunction. reader for more detail of structure of network is encoraged to visit pvae.py.

'''
vocab_size = len(char2ind) 
model = pvae(vocab_size, Args.embedding_size, Args.rnn_type, Args.hidden_size, Args.word_dropout, Args.latent_size,
                    sos_indx, eos_indx, pad_indx, max_sequence_length,Args.nr_classes,Args.device_id,
                    num_layers=1,bidirectional=False, gpu_exist = Args.gpu_exist)
if torch.cuda.is_available():
    torch.cuda.set_device(Args.device_id)
    model = model.cuda(Args.device_id)
    
class_weight = char_weight(txt,char2ind)
class_weight = torch.FloatTensor(class_weight).cuda(device=Args.device_id)

optimizer = torch.optim.Adam(model.parameters(), lr = Args.learning_rate)
NLL = torch.nn.CrossEntropyLoss( weight = class_weight, reduction= 'sum', ignore_index=pad_indx)



step = 0
from loss_vae import *
tracker = defaultdict(list)
for epoch in range(0, Args.epochs):
    
    optimizer,lr = adjust_learning_rate(optimizer, epoch, Args.learning_rate)
    temp = defaultdict(list)
    score_acc = []
    AUC_acc = []
    
    for iteration, batch in enumerate(dataloader_train):
        
        batch = batch2tensor(batch,Args)
        batch_size = batch['input'].size(0)
        model.train()
        ######     Forward pass  ######
        logp, mean, logv, z,pr = model(batch['input'], batch['length'])
       
       
        NLL_loss, KL_loss, KL_weight = loss_fn(NLL,logp, batch['target'],batch['length'], mean, logv, Args.anneal_function, step, Args.k0,Args.x0)
        loss = (NLL_loss + KL_weight * KL_loss )/batch_size
   
        #### Evaluation #####  
        temp['NLL_loss'].append(NLL_loss.item()/batch_size)
        temp['KL_loss'].append(KL_weight*KL_loss.item()/batch_size)
        temp['ELBO'].append(loss.item())
        
        #### Backward pass #####

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        step += 1 # for anneling
   # for iteration, batch in enumerate(dataloader_test):    
    #    batch = batch2tensor(batch,Args)
     #   batch_size = batch['input'].size(0)
      #  model.train()
        ######     Forward pass  ######
       # logp, mean, logv, z,prediction = model(batch['input'], batch['length'])
        #model.eval()

    tracker['NLL_loss'].append(np.mean(np.asarray(temp['NLL_loss'])))
    tracker['KL_loss'].append(np.mean(np.asarray(temp['KL_loss'])))
    tracker['ELBO'].append(np.mean(np.asarray(temp['ELBO'])))
    print(" epoch %04d/%i, 'ELBO' %9.4f, NLL-Loss %9.4f, KL-Loss %9.4f, KL-Weight %6.3f, lr %9.7f"
          %(epoch,Args.epochs, 
            np.mean(np.asarray(tracker['ELBO'])),
            np.mean(np.asarray(tracker['NLL_loss'])),
            np.mean(np.asarray(tracker['KL_loss'])),    
            KL_weight,
            lr))
   
    
    if epoch % Args.save_every == 0:
   
       
        checkpoint_path = os.path.join(Args.save_dir, "E%i.pytorch"%(epoch))
        torch.save(model.state_dict(), checkpoint_path)
save_loss_path = os.path.join(Args.save_dir, "tracker.pickle")
with open(save_loss_path, 'wb') as f:
     pickle.dump(tracker,f)       


