import numpy as np
import sys
from sklearn import metrics
from sklearn import preprocessing
from sklearn.metrics import roc_auc_score
from sklearn.preprocessing import StandardScaler
import torch
import torch.nn as nn
import torch.nn.functional as F
from aggregators import MeanAggregator,ConcatAggregator 
from sklearn.metrics import roc_auc_score
from sklearn.metrics import recall_score




def read_train(filename1,filename2):

   hyperedgelabel = []
   hyperedges = []
   file = open(filename1)
   for line in file:
       ids = line.strip().split()
       hyperedge = []
       if len(ids) < 1:
           continue
       for j in range(len(ids)-1):
           id = ids[j]
           hyperedge.append(int(id)) 
       hyperedgelabel.append(0)
       hyperedges.append(hyperedge)
   true_size = len(hyperedges) * 2
   file2 = open(filename2)
   for line in file2:
       ids = line.strip().split()
       hyperedge = []
       if len(ids) < 1:
           continue
       for j in range(len(ids)):
           id = ids[j]
           hyperedge.append(int(id)) 
       hyperedgelabel.append(1)
       hyperedges.append(hyperedge)
       if len(hyperedges) >= true_size:
           break
  
   return hyperedges, hyperedgelabel
    

def calVar(embeddings,args, test_x) :
    #train = train_x.tolist()
    scaler=preprocessing.StandardScaler()
    embeddings=scaler.fit_transform(embeddings)
    directory = '../data/' + args.dataset + '/'
    train,label_x = read_train(directory + 'trainp.txt',directory + 'tn.txt')
    #print(train)
    #print(label_x)

    h_x_train = []
    e_size = len(embeddings[0])
    for nodes in train:
        h_size = len(nodes)
        tmp_h = []
        for i in range(e_size):
            tmp = []
            for j in range(h_size):
                nj = nodes[j]
                if nj >= len(embeddings):
                    x = [0.0 for i in range(e_size)]
                else:
                    x = embeddings[nj]
                tmp.append(x[i])
            #print(tmp,"nds")
            #tmp_h.append(np.mean(tmp,dtype=np.float64))
            tmp_h.append(np.var(tmp,dtype=np.float64))
            #print(nodes,tmp_h)
        h_x_train.append(tmp_h)
    test = test_x.tolist()
    h_x_test = []
    for l in range(len(test)):
        jj = test[l]
        nodes = jj[0]
        h_size = len(nodes)
        tmp_h = []
        for i in range(e_size):
            tmp = []
            for j in range(h_size):
                nj = nodes[j]
                if nj >= len(embeddings):
                    x = [0.0 for i in range(e_size)]
                else:
                    x = embeddings[nj]
                tmp.append(x[i])
            #print(tmp,"nds")
            tmp_h.append(np.var(tmp,dtype=np.float64))
            #print(nodes,tmp_h)
        h_x_test.append(tmp_h)
    return h_x_train,h_x_test,label_x


def preprocc(X_train,Y_train,X_test,Y_test):
        print("embedding dim: ",len(X_train[0]))
        print(len(X_train),len(Y_train),len(X_test),len(Y_test))
        Y_train = np.array(Y_train, dtype=np.float32)
        scaler=preprocessing.StandardScaler()
        x_train=scaler.fit_transform(X_train)
        x_train=torch.from_numpy(x_train.astype(np.float32))
        x_test=scaler.fit_transform(X_test)
        x_test=torch.from_numpy(x_test.astype(np.float32))
        y_train=torch.from_numpy(Y_train.astype(np.float32))
        
        y_train=y_train.view(y_train.shape[0],1)

        #y_train = torch.LongTensor(Y_train)
        #y_test=torch.from_numpy(Y_test.astype(np.float32))
        y_test=Y_test.float().view(Y_test.shape[0],1)
        #print(X_train[0],len(X_train[0]))
        model=Logistic_Reg_model(len(X_train[0]))
        #model=Cross_Entropy_model(len(X_train[0]),n_types)
        criterion=torch.nn.BCELoss()
        #criterion=torch.nn.CrossEntropyLoss()
        optimizer=torch.optim.Adam(model.parameters(),lr=0.01)
        number_of_epochs=50
        auc = 0.
        recall = 0.
        roc = 0.
        best = []
        for epoch in range(number_of_epochs):
            y_prediction=model(x_train)
            loss=criterion(y_prediction,y_train)
            loss.backward()
            optimizer.step()
            optimizer.zero_grad()
            if (epoch+1)%10 == 0:
                print('epoch:', epoch+1,',loss=',loss.item())
            with torch.no_grad():
                y_pred=model(x_test)
                y_pred_class=y_pred.round()
                accuracy=(y_pred_class.eq(y_test).sum())/float(y_test.shape[0])
                auc = roc_auc_score(y_test,y_pred)
                best.append(auc)
        print("Hyperedge prediction results AUC: ",np.amax(auc))

class Cross_Entropy_model(nn.Module): 
    def __init__(self,no_input_features,n_types):
        super(Cross_Entropy_model,self).__init__()
        self.layer1=torch.nn.Linear(no_input_features,n_types)
        #self.layer2=torch.nn.Linear(16,1)
    def forward(self,x):
        y_predicted=self.layer1(x)
        #y_predicted=torch.sigmoid(self.layer2(y_predicted))

        return y_predicted

class Logistic_Reg_model(nn.Module): 
    def __init__(self,no_input_features):
        super(Logistic_Reg_model,self).__init__()
        self.layer1=torch.nn.Linear(no_input_features,16)
        self.layer2=torch.nn.Linear(16,1)
    def forward(self,x):
        y_predicted=self.layer1(x)
        y_predicted=torch.sigmoid(self.layer2(y_predicted))
        return y_predicted
