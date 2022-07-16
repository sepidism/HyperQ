import torch
import numpy as np
from collections import defaultdict
from model import polyHype
from sklearn.metrics import roc_auc_score
from predict import preprocc
from predict import calVar
args = None

def train(model_args, data):
    global args, model, sess
    args = model_args

    #hedge_and_l, n_types, neighbor_params,hedge_l2,neighbor_params2 = data
    hedge_and_l, n_types, neighbor_params = data
    nnodes = neighbor_params[0]

    train_hedge_l, valid_hedge_l, test_hedge_l = hedge_and_l
    train_size = len(train_hedge_l)

    train_list = []
    train_l = []
    for nodes in train_hedge_l:
        n = nodes[:len(nodes)-1]
        train_list.append(n)
        train_l.append(nodes[len(nodes)-1])
    trains = train_list
    valid_list = []
    for nodes in valid_hedge_l:
        n = nodes[:len(nodes)-1]
        valid_list.append(n)
    test_list = []
    tests = test_list
    test_l = []
    for nodes in test_hedge_l:
        n = nodes[:len(nodes)-1]
        test_list.append(n)
        test_l.append(nodes[len(nodes)-1])

    train_hedges_all = torch.LongTensor(np.array(range(len(train_hedge_l)), np.int32))
    train_hedge = torch.LongTensor(np.array([[t] for t in train_list], np.int32))
    valid_hedge = torch.LongTensor(np.array([[t] for t in valid_list], np.int32))
    test_hedge = torch.LongTensor(np.array([[t] for t in test_list], np.int32))
            
    train_labels = torch.LongTensor(np.array([tuples[len(tuples)-1] for tuples in train_hedge_l], np.int32))
    #print(len(
    valid_labels = torch.LongTensor(np.array([tuples[len(tuples)-1] for tuples in valid_hedge_l], np.int32))
    test_labels = torch.LongTensor(np.array([tuples[len(tuples)-1] for tuples in test_hedge_l], np.int32))

    #real_train_labels = torch.LongTensor(neighbor_params[4], np.int32)
           
    model = polyHype(args,n_types,neighbor_params)

    optimizer = torch.optim.Adam(
            filter(lambda p: p.requires_grad,model.parameters()),
            lr=args.lr,
            )

    if args.cuda:
        model = model.cuda()
        train_labels = train_labels.cuda()
        valid_labels = valid_labels.cuda()
        test_labels = test_labels.cuda()

        train_hedges_all = train_hedges.cuda()
        train_hedge = train_hedge.cuda()
        valid_hedge = valid_hedge.cuda()
        test_hedge = test_hedge.cuda()

    # prepare for top-k evaluation
    true_types = defaultdict(set)
    for nodes in train_hedge_l:
        h = nodes[:len(nodes)-1]
        h = sorted(h)
        true_types[tuple(h)].add(nodes[len(nodes)-1])
    for nodes in valid_hedge_l:
        h = nodes[:len(nodes)-1]
        h = sorted(h)
        true_types[tuple(h)].add(nodes[len(nodes)-1])
    for nodes in test_hedge_l:
        h = nodes[:len(nodes)-1]
        h = sorted(h)
        true_types[tuple(h)].add(nodes[len(nodes)-1])

    best_valid_acc = 0.0
    final_res = None  # acc, mrr, mr, hit1, hit3, hit5

    print('start training ...')

    for step in range(args.epoch):

        # shuffle training data
        #neg_sample = generate_neg_sample(args,true_types,nnode,n_types,train_hedges_l, neighbor_params[2])
        index = np.arange(len(train_labels))
        np.random.shuffle(index)
        train_hedge = train_hedge[index]
        train_hedges_all = train_hedges_all[index]

        train_labels = train_labels[index]
        # training
        s = 0

        while s + args.batch_size <= len(train_labels):
            loss = model.train_step(model, optimizer, get_feed_dict(
                train_hedge, train_hedges_all, train_labels, s, s + args.batch_size))
            s += args.batch_size
        
        

        train_acc, _,train_true,train_pred = evaluate(train_hedge, train_labels)
        print('Classification framework train acc: %.4f' % (train_acc))
    #embeddings = node_embedding(args,train_hedge,train_labels,nnodes,model)
    #partition = (args,test_hedge)
    filename1 = '../data/' + args.dataset + '/' + 'tn.txt'
    filename2 = '../data/' + args.dataset + '/' + 'nodeparts.txt'
    X_train,label_x = find_embedding(args,filename1,trains,train_l,n_types,model)
    #X_train = find_embedding1(args,filename1,train_hedge,train_labels,model)
    X_test,_ = find_embedding2(args,filename2,tests,test_hedge,test_l,n_types,model)
    #embeddings = torch.FloatTensor(embeddings).cuda()
    #X_train, X_test, label_x = calVar(embeddings, args, test_hedge)
    preprocc(X_train,label_x,X_test,test_labels)



def generate_neg_sample(args, true_samples, nnodes, n_train,n_types):

    nodes = [i for i in range(nnodes)]
    hedge = []
    for i in range(n_train):
        h_samp = np.random.choice(list(nodes),size = args.hedge_size)
        h_samp = sorted(h_samp)
        while h_samp in true_samples:
            h_samp = np.random.choice(list(nodes),size = args.hedge_size)
            h_samp = sorted(h_samp)
        h_sampl.append(n_types+1)
        hedge.append(h_samp)
    


def get_feed_dict(train_pairs, train_hedges, labels,start, end):
    feed_dict = {}
    #print(len(train_pairs[start:end]),start,end)
    feed_dict["neighbors"] = train_pairs[start:end]
    if train_hedges is not None:
        feed_dict["train_hedges"] = train_hedges[start:end]

    else:
            # for evaluation no edges should be masked out
        
        feed_dict["train_hedges"] = torch.LongTensor(np.array([-1] * (end - start), np.int32)).cuda() if args.cuda \
                else torch.LongTensor(np.array([-1] * (end - start), np.int32))

    feed_dict["labels"] = labels[start:end]

    return feed_dict


def evaluate(hyperedges, labels):

    acc_list = []
    scores_list = []
    y_true = []
    y_pred = []
    embedding = []


    s = 0
    while s + args.batch_size <= len(labels):
        acc, scores,y_t,y_p,emb = model.test_step(model, get_feed_dict(
            hyperedges, None, labels, s, s + args.batch_size))
        acc_list.extend(acc)
        scores_list.extend(scores)
        y_true.extend(y_t)
        y_pred.extend(y_p)
        s += args.batch_size
        embedding.extend(emb)



    return float(np.mean(acc_list)), np.array(scores_list), y_true, y_pred


def calculate_ranking_metrics(hyperedges_l, scores, true_types):
    for i in range(scores.shape[0]):
        nodes_l = hyperedges_l[i]
        nodes = nodes_l[:len(nodes_l)-1]
        sorted(nodes)
        t = nodes_l[len(nodes_l)-1]
        for j in true_types[tuple(nodes)] - {t}:
            scores[i, j] -= 1.0

    sorted_indices = np.argsort(-scores, axis=1)
    relations = np.array(hyperedges_l)[0:scores.shape[0], 3]
    sorted_indices -= np.expand_dims(relations, 1)
    zero_coordinates = np.argwhere(sorted_indices == 0)
    rankings = zero_coordinates[:, 1] + 1

    mrr = float(np.mean(1 / rankings))
    mr = float(np.mean(rankings))
    hit1 = float(np.mean(rankings <= 1))
    hit3 = float(np.mean(rankings <= 3))
    hit5 = float(np.mean(rankings <= 5))

    return mrr, mr, hit1, hit3, hit5

def node_embedding(args,hyperedges,labels,nnodes,model):

    embedding = []


    s = 0
    batch_size = 1
    while s + args.batch_size <= len(labels):
        _, _,_,_,emb = model.test_step(model, get_feed_dict(
            hyperedges, None, labels, s, s + args.batch_size))
        s += args.batch_size
        embedding.extend(emb)
    ss = len(embedding[0])
    Embedding = torch.FloatTensor(embedding).cuda() if args.cuda else torch.FloatTensor(embedding)
    Embedding = Embedding.unsqueeze(-1)
    x = torch.permute(Embedding,(0,2,1))
    Embedding = torch.bmm(Embedding,x)
    Embedding = Embedding.view([-1, ss*ss])
    embw = open("nodeEmbedding.txt","w")

    NodeEmb = []#torch.zeros([nsize,ss*ss],dtype=torch.float64)
    for n in range(len(nnodes)):
        neighbor_hedges = nnodes[n]
        neighbor_hedges = torch.LongTensor(neighbor_hedges)
        neighbor_emb = torch.index_select(Embedding, 0,neighbor_hedges) 
        neighbor_emb = torch.mean(neighbor_emb,dim=-2).tolist()

        NodeEmb.append(neighbor_emb)
    f = open("iJbin.txt","r")
    i = 0
    for line in f:
        w = line.split()
        tmp = []
        for a in w:
            val = float(a)
            tmp.append(val)
        NodeEmb[i].extend(tmp)
        i += 1

    for i in range(len(NodeEmb)):
        #embw.write(str(i))
        #embw.write(" ")
        for e in NodeEmb[i]:
            embw.write(str(e))
            embw.write(" ")
        embw.write("\n")


    return NodeEmb

def find_embedding1(args,filename,train,labels,model):

    embedding = []
    hyperedges = []
    

    s = 0
    batch_size = 1
    while s + args.batch_size <= len(labels):
        _, _,_,_,emb = model.test_step(model, get_feed_dict(
            train, None, labels, s, s + args.batch_size))
        s += args.batch_size
        embedding.extend(emb)
    ss = len(embedding[0])
    Embedding = torch.FloatTensor(embedding).cuda() if args.cuda else torch.FloatTensor(embedding)
    Embedding = Embedding.view([-1, ss])

    return Embedding

def find_embedding(args,filename,train,labels,label_size,model):

    embedding = []
    hyperedges = []
    binlabel = [0 for i in range(len(labels))]
    
    f = open(filename,'r')
    for line in f:
       ids = line.strip().split()
       hyperedge = []
       if len(ids) < 1:
           continue
       for j in range(len(ids)):
           id = ids[j]
           hyperedge.append(int(id)) 
       labels.append(label_size-1)
       hyperedges.append(hyperedge)
       binlabel.append(1)
    for newn in hyperedges:
        
        sampled=np.random.choice(list(newn),size=args.hedge_size,\
                replace=len(newn) < args.hedge_size)
        train.append(sampled)

    hyperedgess = torch.LongTensor(np.array([[t] for t in train], np.int32))
    labels = torch.LongTensor(labels)
    s = 0
    batch_size = 1
    while s + args.batch_size <= len(labels):
        _, _,_,_,emb = model.test_step(model, get_feed_dict(
            hyperedgess, None, labels, s, s + args.batch_size))
        s += args.batch_size
        embedding.extend(emb)
    ss = len(embedding[0])
    Embedding = torch.FloatTensor(embedding).cuda() if args.cuda else torch.FloatTensor(embedding)
    Embedding = Embedding.view([-1, ss])

    return Embedding,binlabel

def find_embedding2(args,filename,test,test_torch,labels,label_size,model):

    embedding = []
    hyperedges = []
    nodepart = []
    newlabels = []
    types = {}
    f = open(filename,'r')
    for line in f:
        w = line.split()
        node = int(w[0])
        part = int(w[1])
        types[part] = 1
        nodepart.append(part)
    
    for i in range(len(test)):
        if labels[i] == 1:
            newlabels.append(label_size-1)
            continue
        pp = [0 for i in range(label_size)]
        newn = test[i]
        for n in newn:
            p = nodepart[n]
            pp[p] += 1
        idm = 0;
        pmax = 0
        for i in range(len(pp)):
            if pp[i] > pmax:
                idm = i
                pmax = pp[i]
        newlabels.append(idm)

    tlabels = torch.LongTensor(newlabels)
    s = 0
    batch_size = 1
    t_size = len(types)
    #if args.prediction == True:
    while s + args.batch_size <= len(labels):
        _, _,_,_,emb = model.test_step(model, get_feed_dict(
            test_torch, None, tlabels, s, s + args.batch_size))
        s += args.batch_size
        embedding.extend(emb)

    #print(embedding)
    ss = len(embedding[0])
    Embedding = torch.FloatTensor(embedding).cuda() if args.cuda else torch.FloatTensor(embedding)
    #Embedding = Embedding.unsqueeze(-1)
    #x = torch.permute(Embedding,(0,2,1))
    #Embedding = torch.bmm(Embedding,x)
    #Embedding = Embedding.view([-1, ss*ss])
    Embedding = Embedding.view([-1, ss])
    print(Embedding.shape,"test shape")

    return Embedding,tlabels

