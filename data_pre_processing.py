# -*- coding: utf-8 -*-
"""
Created on Wed Nov 23 16:21:17 2022

@author: Administrator
"""

import pandas as pd
import pickle
import re
small_test = pd.read_csv("..\data\csvList\small_test.csv")
small_train = pd.read_csv("..\data\csvList\small_train.csv")

proj = small_train['proj'].tolist()
files = small_train['filename'].tolist()
paths = [["..\data\small_train\\"+i+"\\"+j+'_extract.pkl',j.split("-")[-2]] for i,j in zip(proj,files)]




#fid groupid opt code
name_dict = {}
count = 0
count_group = 0
f2 = open("feat.csv","w",encoding='UTF-8')
path_count = 0

for path in paths:
    path_count = path_count+1
    try:
        f = open(path[0],'rb')
    
        data=pickle.load(f)
        names = list(data.keys())
        for name in names:
            count = count+1
            
            if name_dict.get(name,'1')=='1':
                #name_dict[name] = count_group
                count_group = count_group+1
                name_dict[name] = count_group
                line =  str(count)+"\t"+ str(name_dict[name])+"\t"+path[1]+"\t"+"~~".join([i.replace("    ",",") for i in data[name][1]]).replace('\xa9','')+"\n"
                line = re.sub("[\u4e00-\u9fa5\u0800-\u4e00\uac00-\ud7ff]", '', line)
                f2.write(line)
            else:
                line = str(count)+"\t"+ str(name_dict[name])+"\t"+path[1]+"\t"+"~~".join([i.replace("    ",",") for i in data[name][1]]).replace('\xa9','')+"\n"
                line = re.sub("[\u4e00-\u9fa5\u0800-\u4e00\uac00-\ud7ff]", '', line)
                f2.write(line)
                name_dict[name] = name_dict[name] +1
            
            
            
            if count%10000 == 0:
                print(path_count)
    except FileNotFoundError:
        print(path_count)
f2.close()
#feat = pd.DataFrame(feat,columns=['fid','groupid','opt','code'])

