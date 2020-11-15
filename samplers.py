""" Sampler for dataloader. """
import torch
import numpy as np
import random
from itertools import combinations 
# Customize such as total way number of distinct classes to segment in a meta task

def nCr(n, r): 
  
    return (fact(n) / (fact(r)  
                * fact(n - r))) 
  
# Returns factorial of n 
def fact(n): 
  
    res = 1
      
    for i in range(2, n+1): 
        res = res * i 
          
    return res 
class CategoriesSampler():
    """The class to generate episodic data"""
    def __init__(self, labeln, unique_labels, n_batch, K, N, Q):
        #K Way, N shot(train query), Q(test query)
        self.unique_labels = unique_labels.tolist()
        
        self.K = K
        self.N = N
        self.Q = Q
        # print(unique_labels)
        comb = combinations(self.unique_labels, K)
        self.label_combos = [c for c in comb]
        # print(self.label_combos)
        self.n_batch = len(self.label_combos)
        print("NUM batches = {}".format(self.n_batch))
        # labeln = np.array(labeln)
        self.m_ind = {}
        for i in self.unique_labels:
            ind = np.argwhere(labeln == i).reshape(-1)
            ind = torch.from_numpy(ind)
            self.m_ind[i] = ind
        self.index = 0
    def __len__(self):
        return self.n_batch
    
    def __iter__(self):
        for i in range(self.n_batch):
            classes = self.label_combos[self.index]
            self.index += 1
            if self.index >= self.n_batch:
                self.index -= self.n_batch
            # print(classes)
            # classes = torch.randperm(len(self.m_ind))[:(self.K-1)]
            lr=[]
            dr=[]
            for c in classes:
                # c_ind = self.unique_labels.index(c)
                l = self.m_ind[c]
                pos = torch.randperm(len(l))[:(self.N +self.Q)]
                m=l[pos]
                
                for i in range(0,self.N):
                    lr.append(m[i])
                    
                for i in range(self.N, (self.N +self.Q)):
                    dr.append(m[i])
            
            # # redundancy for background
            # h=random.randint(0,self.K-2)
            # c=classes[h]
            # l = self.m_ind[c]
            # pos = torch.randperm(len(l))[:(self.N +self.Q)]
            # m=l[pos]

            # for i in range(0,self.N):
            #     lr.append(m[i])

            # for i in range(self.N, (self.N +self.Q)):
            #     dr.append(m[i])   

            batch=[]
            for i in range(len(lr)):
                batch.append(lr[i])
            
            for i in range(len(dr)):
                batch.append(dr[i])
                        
            batch = torch.stack(batch).t().reshape(-1)      
                
            yield batch



class ValCategoriesSampler():
    """The class to generate episodic data"""
    def __init__(self, labeln, unique_labels, label_combos, N, Q):
        #K Way, N shot(train query), Q(test query)
        
        self.n_batch = len(label_combos)
        self.label_combos = label_combos
        self.N = N
        self.Q = Q

        # labeln = np.array(labeln)
        self.m_ind = {}
        for i in unique_labels:
            ind = np.argwhere(labeln == i).reshape(-1)
            ind = torch.from_numpy(ind)
            self.m_ind[i] = ind

    def __len__(self):
        return self.n_batch
    
    def __iter__(self):
        for i_batch in range(self.n_batch):
            classes = self.label_combos[i_batch]
            lr=[]
            dr=[]
            for c in classes:
                l = self.m_ind[c]
                pos = torch.randperm(len(l))[:(self.N +self.Q)]
                m=l[pos]
                
                for i in range(0,self.N):
                    lr.append(m[i])
                    
                for i in range(self.N, (self.N +self.Q)):
                    dr.append(m[i])
            
            # # redundancy for background
            # h=random.randint(0,self.K-2)
            # c=classes[h]
            # l = self.m_ind[c]
            # pos = torch.randperm(len(l))[:(self.N +self.Q)]
            # m=l[pos]

            # for i in range(0,self.N):
            #     lr.append(m[i])

            # for i in range(self.N, (self.N +self.Q)):
            #     dr.append(m[i])   

            batch=[]
            for i in range(len(lr)):
                batch.append(lr[i])
            
            for i in range(len(dr)):
                batch.append(dr[i])
                        
            batch = torch.stack(batch).t().reshape(-1)      
                
            yield batch
