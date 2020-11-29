import torch
from torch.utils.data.sampler import Sampler
import numpy as np
import time
import pickle  

class mySampler(Sampler):
    def __init__(self,data_source, batchsize, ids_num_per_batch):
        # batchsize=batchsize1/3
        self.data_source = data_source
        self.batchsize = batchsize
        self.iters_per_epoch = len(data_source)//batchsize
        # self.labels_list = labels_list
        self.ids_num_per_batch = ids_num_per_batch
        self.img_per_id = batchsize//ids_num_per_batch
        # get id dictionary (2d), each ids has xxx imgs
        # self.id_dic = {}
        # since = time.time()
        # for i in range(len(data_source)):
        #     if i % 10000 == 0:
        #         print(['processing  ' + str(i) + '  images !!!!!!!!!'])
        #         time_elapsed = time.time() - since
        #         time_left = time_elapsed / (i+1) * len(data_source)
        #         print('used time : {:.0f}m {:.0f}s'.format(time_elapsed // 60, time_elapsed % 60))
        #         print('left time : {:.0f}m {:.0f}s'.format(time_left // 60, time_left % 60))
        #     # if i % 1000000 == 0:
        #         # self._save_obj(self.id_dic, 'id_dic_' + str(i)) 
        #     one_id = data_source[i][1]
        #     self.id_dic.setdefault(one_id,[]) 
        #     self.id_dic[one_id].append(i)
        # # self._save_obj(self.id_dic, 'id_dic')
        self.id_dic = self._load_obj('id_dic')
        self.id_num = len(self.id_dic)
        # print(self.id_dic)
    def __len__(self):
        return self.iters_per_epoch*self.batchsize

    def _save_obj(self, obj, name ):
        with open('obj_sysu30k/'+ name + '.pkl', 'wb') as f:
            pickle.dump(obj, f, pickle.HIGHEST_PROTOCOL)

    def _load_obj(self, name ):
        with open('obj_sysu30k/' + name + '.pkl', 'rb') as f:
            return pickle.load(f)
    def _get_index(self, theSet, theNumber):
        if len(theSet) >= theNumber:
            tt = np.random.choice(theSet, size=theNumber, replace=False) 
        else:
            tt = np.random.choice(theSet, size=theNumber, replace=True)
        return tt
    def __iter__(self):
        # select ids_num_per_batch
        ret = []
        mode = np.asarray([0,1,2])
        mode = np.tile(mode, [self.ids_num_per_batch // 3])
        mode = np.expand_dims(mode, 0)
        mode = np.tile(mode, [self.iters_per_epoch, 1])
        for i in range(self.iters_per_epoch):
            t = torch.randperm(self.id_num) 
            for j in range(self.ids_num_per_batch):
                idx = t[j]
                idx_l1 = max(t[j]-1, 0)
                idx_l2 = max(t[j]-2, 0)
                idx_r1 = min(t[j]+1, self.id_num-1)
                idx_r2 = min(t[j]+2, self.id_num-1)
                if mode[i][j] == 0:
                    tt = self._get_index(self.id_dic[idx], self.img_per_id)
                elif mode[i][j] == 1:
                    t1 = self._get_index(self.id_dic[idx_l1], self.img_per_id // 3)
                    t2 = self._get_index(self.id_dic[idx], self.img_per_id // 3)
                    t3 = self._get_index(self.id_dic[idx_r1], self.img_per_id - self.img_per_id // 3 * 2)
                    tt = np.concatenate((t1, t2, t3), axis = 0)
                else:
                    t1 = self._get_index(self.id_dic[idx_l1], self.img_per_id // 5)
                    t2 = self._get_index(self.id_dic[idx_l2], self.img_per_id // 5)
                    t3 = self._get_index(self.id_dic[idx], self.img_per_id // 5)
                    t4 = self._get_index(self.id_dic[idx_r1], self.img_per_id // 5)
                    t5 = self._get_index(self.id_dic[idx_r2], self.img_per_id - self.img_per_id // 5 * 4)
                    tt = np.concatenate((t1, t2, t3, t4, t5), axis = 0)
                ret.extend(tt) 
        return iter(ret)

