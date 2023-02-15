# -*- coding: utf-8 -*-
"""
Created on Thu Nov 24 14:25:10 2022

@author: 18523
"""

import pandas as pd
import json
import random
with open("feat.csv","r") as f:
    feat = f.readlines()

feat = [i.split("\t") for i in feat]
feat = pd.DataFrame(feat)
feat.columns = ['fid','groupid','opt','code']
feat["group_count"] = feat.groupby('groupid')["groupid"].transform('count')
feat = feat.loc[(feat.group_count<=5) & (feat.group_count>=2)].reset_index(drop=True)

"""
feat['code'] = feat.apply(lambda x:
            x['code'].strip().replace(" ",
            "").replace("\"",
            "").replace("'","").replace("::",",").replace(":",
            ",").replace("*","").replace("+",",+,").replace("[",
            "").replace("]","").replace(";",",").replace("_",","), 
            axis=1)  
"""

feat['code'] = feat.apply(lambda x:
            x['code'].strip().replace(" ","").replace(";",",").replace("\"","").replace("'","").replace("_",",").replace("::",","), 
            axis=1)   
fid_code = {k:v for k,v in zip(feat["fid"].tolist(),feat["code"].tolist())}
with open("fid_code.json", "w") as f:
    f.write(json.dumps(fid_code))



count = 0
token_dict ={}
for code in feat.code.tolist():
    count = count+1
    if count%10000==0:
        print(count)
    code = code.replace("~~",",")
    code = code.split(",")
      
    code = set(code)
    code = [i for i in code if i.isalnum()]
    #code = [i for i in code if i[-1]!='h']
    codes = [i for i in code if i[0].isdigit()!=True]
    
    """
    codes = []
    for i in code:
        if i[0].isdigit()!=True:
            codes=codes+[i]
        else:
            try:
                Operands = str(int(i[:-1],16)%300)
                codes=codes+[Operands]
            except ValueError:
                flag=1    
    """
    
    
    for token in codes:
        if token_dict.get(token,'-5')=='-5':
            token_dict[token] = 0
        else:
            token_dict[token] = token_dict[token]+1


token_dict = {k:v for k,v in token_dict.items() if v>32}
token_dict = {k:v for k,v in zip(token_dict.keys(),range(len(token_dict))) }


with open("token_dict.json", "w") as f:
    f.write(json.dumps(token_dict))


#O0,O3 O1,O3 O2,O3 O0,Os O1,Os O2,Os
id_feat = feat[['fid','groupid','opt']]
id_feat['opt'] = id_feat['opt'].astype(str)
id_feat = id_feat.sort_values(['groupid']).reset_index(drop=True)
id_feat['rindex'] = id_feat.index.tolist()



id_feat = list(id_feat.groupby('groupid'))
id_feat = [[i[0],i[1]['fid'].tolist(),i[1]['opt'].tolist()] for i in id_feat]

random.seed(2022) 
random.shuffle(id_feat)

count = 0
count_sample = 0
max_count = len(id_feat)
sample = []
file = open("sample.csv","w")
nage_rate = 30
for feat in id_feat:
    count = count+1
    lenth = len(feat[1])
    if count+110>max_count:
        count = 0
    #if count>max_count:
    #    break

    
    if lenth==2:        
        
        if fid_code[feat[1][0]] !=fid_code[feat[1][1]]:            
            sample = sample + [feat[1]+[1]] 
            
            sample = sample+[[feat[1][0],i[1][0],-1] for i in id_feat[count:count+nage_rate]]
            sample = sample+[[feat[1][1],i[1][0],-1] for i in id_feat[count+nage_rate:count+nage_rate*2]]            
        else:
            sample = sample+[[feat[1][1],i[1][0],-1] for i in id_feat[count:count+nage_rate]]   
        

    elif lenth==3:    
        
        
        
        nage_samp = [i[1][0] for i in id_feat[count:count+nage_rate*3]]
        ncont = 0
        indexs = 0
        for sap_index in [[0,1],[1,2],[0,2]]:
            if fid_code[feat[1][sap_index[0]]] !=fid_code[feat[1][sap_index[1]]]:       
                
                sample = sample + [[feat[1][sap_index[0]],feat[1][sap_index[1]],1]]
            
            if indexs<3:
                sample = sample + [[feat[1][indexs],i,-1] for i in nage_samp[nage_rate*ncont:nage_rate*(ncont+1)]]
                ncont=ncont+1
                indexs = indexs+1
    
    elif lenth==4:    
        nage_samp = [i[1][0] for i in id_feat[count:count+nage_rate*4]]   
        ncont = 0
        indexs = 0
        for sap_index in [[0,1],[1,2],[2,3],[0,2],[0,3],[1,3]]:
            
            if fid_code[feat[1][sap_index[0]]] !=fid_code[feat[1][sap_index[1]]]:       
                
                sample = sample + [[feat[1][sap_index[0]],feat[1][sap_index[1]],1]]
            
            if indexs<4:
                sample = sample + [[feat[1][indexs],i,-1] for i in nage_samp[nage_rate*ncont:nage_rate*(ncont+1)]]
                ncont=ncont+1
                indexs = indexs+1
                    
         
    elif lenth==5:    
        nage_samp = [i[1][0] for i in id_feat[count:count+nage_rate*5]]       
        ncont = 0
        indexs = 0
        for sap_index in [[0,1],[0,2],[0,3],[0,4],[1,2],[1,3],[1,4],[2,3],[2,4],[3,4]]:
            
            if fid_code[feat[1][sap_index[0]]] !=fid_code[feat[1][sap_index[1]]]:       
                
                sample = sample + [[feat[1][sap_index[0]],feat[1][sap_index[1]],1]]
            
            if indexs<5:
                sample = sample + [[feat[1][indexs],i,-1] for i in nage_samp[nage_rate*ncont:nage_rate*(ncont+1)]]
                ncont=ncont+1
                indexs = indexs+1   

    if len(sample)>50000:
        print(count_sample)
        count_sample =count_sample+1
        for item in sample:       
            item[-1] = str(item[-1])
            file.write(','.join(item)+"\n")
        del sample
        sample = []
for item in sample:
    item[-1] = str(item[-1])            
    file.write(','.join(item)+"\n")
file.close()

