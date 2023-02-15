import json
import numpy as np
import torch
from torch.utils.data import Dataset, DataLoader
import time
import pandas as pd
from prefetch_generator import BackgroundGenerator
from model import code_text_cnn,code_MLPMixer,cross_code_transformer,cross_code_lstm
from tqdm import tqdm
import pickle


"""
with open('test_code.pkl', 'rb') as f:
    test_code = pickle.load(f) 
"""    
with open('orage_code.pkl', 'rb') as f:
    orage_code = pickle.load(f)     
    

"""
   # orage_code
with open('big_orage_code.pkl', 'rb') as f:
    orage_code = pickle.load(f)     
"""    
    
#['O0', 'O1', 'O2', 'O3', 'Os']  

test_code = [{j:k for j,k in zip(['O0', 'O1', 'O2', 'O3', 'Os'],i)} for i in orage_code]


def eval_O(ebds,TYPE1,TYPE2,POOLSIZE):
    funcarr1=[]
    funcarr2=[]

    for i in range(len(ebds)):
        if ebds[i].get(TYPE1) is not None and type(ebds[i][TYPE1]) is not int:
            if ebds[i].get(TYPE2) is not None and type(ebds[i][TYPE2]) is not int:
                ebd1,ebd2=ebds[i][TYPE1],ebds[i][TYPE2]
                funcarr1.append(ebd1 / ebd1.norm())
                funcarr2.append(ebd2 / ebd2.norm())
        else:
            continue

    ft_valid_dataset=FunctionDataset_Fast(funcarr1,funcarr2)
    dataloader = DataLoader(ft_valid_dataset, batch_size=POOLSIZE, shuffle=True)
    SIMS=[]
    Recall_AT_1=[]

    for idx, (anchor,pos) in enumerate(tqdm(dataloader)):
        anchor = anchor.cuda()
        pos =pos.cuda()
        if anchor.shape[0]==POOLSIZE:
            for i in range(len(anchor)):    # check every vector of (vA,vB)
                vA=anchor[i:i+1]  #pos[i]
                sim = np.array(torch.mm(vA, pos.T).cpu().squeeze())
                y=np.argsort(-sim)
                posi=0
                for j in range(len(pos)):
                    if y[j]==i:
                        posi=j+1
                        break 
                if posi==1:
                    Recall_AT_1.append(1)
                else:
                    Recall_AT_1.append(0)
                SIMS.append(1.0/posi)
    print(TYPE1,TYPE2,'MRR{}: '.format(POOLSIZE),np.array(SIMS).mean())
    print(TYPE1,TYPE2,'Recall@1: ', np.array(Recall_AT_1).mean())
    
    #with open("log_att_bg2.txt", "a") as f:
    #     f.write('MRR{}: '.format(POOLSIZE)+str(np.array(SIMS).mean())+"\t"+'Recall@1: '+str(np.array(Recall_AT_1).mean())+"\n")   
         #f.write('Recall@1: '+str(np.array(Recall_AT_1).mean())+"\n")   
         
    return np.array(Recall_AT_1).mean()

class FunctionDataset_Fast(torch.utils.data.Dataset): 
    def __init__(self,arr1,arr2): 
        self.arr1=arr1
        self.arr2=arr2
        assert(len(arr1)==len(arr2))
    def __getitem__(self, idx):            
        return self.arr1[idx].squeeze(0),self.arr2[idx].squeeze(0)
    def __len__(self):
        return len(self.arr1)


def test(model,is_all):
    model.eval()
    code_vector = []
    count = 0
    start = time.time()
    for code_dict in test_code:
        count = count+1
        if count%2000==0:
            #print(count)
            #print(time.time()-start)
            if is_all!=True:
                break
        vector = {}
        for key in list(code_dict.keys()):
            token_ids = []
            padsize=256
            tokens = code_dict[key]
            tokens = tokens.split("~~")
            tokens = [i.replace("+",",+,").replace("[","").replace("]","").split(",") for i in tokens]
            
            for i in tokens:
                token_id = []
                for j in i:
                    if len(token_id)<=5:
                        if token2id_dict.get(j,'-1')!='-1':
                            token_id = token_id+ [token2id_dict[j]]
                            
                if len(token_id)<=5:
                    token_id = token_id+[len(token2id_dict)]*(5-len(token_id))
                
                if len(token_id)>5:
                    token_id = token_id[:5]
                
                token_ids = token_ids+ [token_id]
            
            if len(token_ids)>padsize:
                token_ids = token_ids[:padsize]
            else:
                token_ids = token_ids+[[len(token2id_dict)]*5]*(padsize - len(token_ids))
                
            token_ids = np.array(token_ids)
            token_ids = torch.from_numpy(token_ids).type(torch.LongTensor)
            token_ids = token_ids.unsqueeze(dim=0)
            
            
            position = [i for i in range(padsize)]
            position = np.array(position)
            position = torch.from_numpy(position).type(torch.LongTensor)
            position = position.unsqueeze(dim=0)

            #token_emb = model.get_code_post_feat(position.cuda(),token_ids.cuda())
            #token_emb,_ = model.get_code_post_feat(token_ids.cuda())
            token_emb,_ = model.get_code_post_feat(position.cuda(),token_ids.cuda())
            vector[key] = token_emb.data.cpu()
        code_vector = code_vector+[vector]
    eval_O(code_vector,'O0','O3',32)
    if is_all:        
        eval_O(code_vector,'O0','Os',32)
        eval_O(code_vector,'O1','Os',32)
        eval_O(code_vector,'O1','O3',32)
        eval_O(code_vector,'O2','Os',32)
        eval_O(code_vector,'O2','O3',32)
        
        
        
        eval_O(code_vector,'O0','O3',10000)
        eval_O(code_vector,'O0','Os',10000)
        eval_O(code_vector,'O1','Os',10000)
        eval_O(code_vector,'O1','O3',10000)
        eval_O(code_vector,'O2','Os',10000)
        eval_O(code_vector,'O2','O3',10000)        
        
        #eval_O(ebds,'O0','Os')
        #eval_O(ebds,'O1','Os')
        #eval_O(ebds,'O1','O3')
        #eval_O(ebds,'O2','Os')
        #eval_O(ebds,'O2','O3')
    #for code in test_code:


class DataLoaderX(DataLoader):
    def __iter__(self):
        return BackgroundGenerator(super().__iter__())



with open("fid_code.json", "r") as f:
    fid_code_map = json.loads(f.read())

    
with open("token_dict.json", "r") as f:
    token2id_dict = json.loads(f.read())
    token2id_dict['+'] = len(token2id_dict)
    token2id_dict['cs:'] = len(token2id_dict)

sample_id = pd.read_csv(r'sample.csv',encoding='utf-8',header=None)
sample_id = sample_id.values

class My_dataset_test(Dataset):
    def __init__(self,sample_id,padsize=256):
        super().__init__()   
    
        self.train_id_label = sample_id
        self.padsize  = 256
        
    def get_token(self, fid):
        
        token = fid_code_map[fid]
        
        tokens = token.split("~~")
        tokens = [i.replace("+",",+,").replace("[","").replace("]","").split(",") for i in tokens]
        
        """
        codes = []
        for i in tokens:
            if i[0].isdigit()!=True:
                codes=codes+[i]
            else:
                try:
                    Operands = str(int(i[:-1],16)%200)
                    codes=codes+[Operands]
                except ValueError:
                    codes=codes+[i]
        
        tokens = codes        
        """
        
        
        token_ids = []
        for i in tokens:
            token_id = []
            for j in i:
                if len(token_id)<=5:
                    if token2id_dict.get(j,'-1')!='-1':
                        token_id = token_id+ [token2id_dict[j]]
                        
            if len(token_id)<=5:
                token_id = token_id+[len(token2id_dict)]*(5-len(token_id))
            
            if len(token_id)>5:
                token_id = token_id[:5]
            
            token_ids = token_ids+ [token_id]
        
        if len(token_ids)>self.padsize:
            token_ids = token_ids[:self.padsize]
        else:
            token_ids = token_ids+[[len(token2id_dict)]*5]*(self.padsize - len(token_ids))
            
        token_ids = np.array(token_ids)
        return token_ids
    
    def __len__(self):
        return len(self.train_id_label)
    
    def __getitem__(self, index):
        
        data = self.train_id_label[index]    
        token_index = self.get_token(str(data[0]))
        token_nage = self.get_token(str(data[1]))
        label = int(str(data[2]))
        position = [i for i in range(self.padsize)]
        position = np.array(position)
        
        return token_index,token_nage,position,label


if __name__ == '__main__':

    data_set =  My_dataset_test(sample_id)
    data_loader = DataLoaderX(data_set, batch_size=2,)#shuffle=True,num_workers=4)   
    #data_loader = DataLoaderX(data_set, batch_size=2)
    
    vocab_size = len(token2id_dict)+1
    padding_idx = len(token2id_dict)
    pad_size = 256
    model = code_text_cnn(vocab_size,padding_idx,pad_size).cuda()
    

    #model = cross_code_lstm(vocab_size,padding_idx,pad_size).cuda()
    #model = code_MLPMixer(vocab_size,padding_idx,pad_size).cuda()
    #model.load_state_dict(torch.load("model_epoch_1.pt"))
    max_epoch = 2
    optimizer = torch.optim.Adam(model.parameters(),lr=1e-3)
    scheduler =  torch.optim.lr_scheduler.CosineAnnealingLR(optimizer = optimizer,T_max =  len(data_loader))
    #loss_fun =  torch.nn.TripletMarginLoss(margin=1)
    loss_fun = torch.nn.CosineEmbeddingLoss(margin=0.9)

    
    loss_s = 0
    step = 0
    loss_p = []
    
    #model.load_state_dict(torch.load("model_epoch_1.pt"))
    n_params = sum(p.numel() for p in model.parameters())
    print("number of parameters: %.2fM" % (n_params/1e6,))    
    print(padding_idx)
    for epoch in range(max_epoch):
        for batch in data_loader:
            
            step = step+1
            token_index,token_nage,position,label= batch[0].type(torch.LongTensor).cuda(),batch[1].type(torch.LongTensor).cuda(),batch[2].type(torch.LongTensor).cuda(),batch[3].type(torch.FloatTensor).cuda()
            
            index_feat,nage_feat = model(position,token_index,token_nage)
            loss = loss_fun(index_feat, nage_feat,label)
        

        
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()  
            scheduler.step()
            fk=loss.cpu()
            loss_s = loss_s+fk.data.numpy()  
            #if step%300==0:
            #    print(loss_s/step,step)
            if step%2000==0 :#and step>10000:
                print(step)
                test(model,False)
                model.train()
                #torch.save(model.state_dict(), f="model_step_{}.pt".format(step))
            if step%4000==0 and step>=10000:
                torch.save(model.state_dict(), f="model_step_{}.pt".format(step))
            #    print(step,optimizer.param_groups[0]['lr'])
                test(model,True)
                model.train()
                
                #loss_p.append(loss_s/step)          
                #with open("log.txt", "a") as f:
                #     f.write(str(loss_s/step)[0:5]+"\n")           
                #torch.save(model.state_dict(), f="model_step_{}.pt".format(step))




    