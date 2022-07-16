import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from aggregators import MeanAggregator,ConcatAggregator 


class polyHype(nn.Module): 
    def __init__(self,args,n_types, params_neighbor):
        super(polyHype, self).__init__()
        self._parse_args(args, n_types, params_neighbor)
        self._build_model()


    def _parse_args(self, args, n_types,params_neighbor):
        self.n_types = n_types
        self.use_gpu = args.cuda

        self.dataset = args.dataset
        self.batch_size = args.batch_size
        self.hidden_dim = args.dim
        self.feature_type = args.feature_type
        self.hedge_size = args.hedge_size
        #self.context_hops = args.context_hops

        self.neighborhedges = torch.LongTensor(params_neighbor[0]).cuda() if args.cuda \
                else torch.LongTensor(params_neighbor[0])
        self.hyperedges = torch.LongTensor(params_neighbor[1]).cuda() if args.cuda  \
                else torch.LongTensor(params_neighbor[1])
        #print(self.hyperedges,"** hedges")
        self.hedgetypes = torch.LongTensor(params_neighbor[2]).cuda() if args.cuda else \
                torch.LongTensor(params_neighbor[2])
        self.nodeEmb = torch.LongTensor(params_neighbor[3]).cuda() if args.cuda else \
                torch.LongTensor(params_neighbor[3])
        self.neighbor_samples = args.neighbor_samples
        self.neighbor_agg = MeanAggregator
        print(self.neighborhedges.shape)
        print(self.hyperedges.shape)
        print(self.hedgetypes.shape)
        print(self.n_types)
        #self.neighbor_agg = ConcatAggregator


    def _build_model(self):
        
        self._build_type_feature()

        self.scores = 0.0


        self.aggregators = nn.ModuleList(self._get_neighbor_aggregators())

    def forward(self, batch):
        self.node_pairs = batch['neighbors']
        self.train_hyperedge = batch['train_hedges']
        self.labels = batch['labels']

        self._call_model()

    def _call_model(self):
        self.scores = 0.
        hedge_list, mask_list, node_emb = self._get_neighbors_and_masks(self.labels, self.node_pairs, self.train_hyperedge)
        self.aggregated_neighbors, self.embedding = self._aggregate_neighbors(hedge_list, mask_list,node_emb)
        self.scores += self.aggregated_neighbors
        self.scores_normalized = torch.sigmoid(self.scores)

    def _build_type_feature(self):
        self.type_dim = self.n_types
        self.type_features = torch.eye(self.n_types).cuda if self.use_gpu \
                else torch.eye(self.n_types)
        
        

    def _get_neighbors_and_masks(self, types, node_pairs, train_hyperedges):
        
        hedge_list = [types]
        masks = []
        train_hyperedges = torch.unsqueeze(train_hyperedges, -1) # this is the edge that we are training

        neighbor_nodes = node_pairs

        neighbor_edges = torch.index_select(self.neighborhedges, 0,neighbor_nodes.view(-1)).view([self.batch_size,-1]) 
        node_emb = torch.index_select(self.nodeEmb, 0,neighbor_nodes.view(-1)).view([self.batch_size,self.hedge_size,-1]) 
        hedge_list.append(neighbor_edges)
        mask = neighbor_edges - train_hyperedges
        mask = (mask != 0).float()
            
        masks.append(mask)
        return hedge_list, masks,node_emb

    def _get_neighbor_aggregators(self):
        aggregators = []
        emb_size = self.nodeEmb.size(dim=-1)
        aggregators.append(self.neighbor_agg(batch_size=self.batch_size,
                                                 input_dim=self.type_dim,
                                                 output_dim=self.n_types))
        return aggregators

    def _aggregate_neighbors(self, hedge_list, mask_list,node_emb):
        edge_vectors = [torch.index_select(self.type_features,0,hedge_list[0])]
        edge_vectors2 = []
        for edges in hedge_list[1:]:
            #print(edges.shape)
            types = torch.index_select(self.hedgetypes,0,edges.view(-1)).view(list(edges.shape)+[-1])
            edge_vectors.append(torch.index_select(self.type_features,0,
                types.view(-1)).view(list(types.shape)+[-1]))
            
        aggregator = self.aggregators[0]
        hedge_vectors_next_iter = []
        neighbors_shape = [self.batch_size,-1,self.hedge_size,self.neighbor_samples, aggregator.input_dim]
        masks_shape = [self.batch_size,-1,self.hedge_size,self.neighbor_samples,1]
        vector1,vector2 = aggregator(self_vectors=edge_vectors[0], \
                    neighbor_vectors=edge_vectors[1].view(neighbors_shape),
                        masks=mask_list[0].view(masks_shape),node_emb=node_emb)
        hedge_vectors_next_iter.append(vector1)                
        edge_vectors=hedge_vectors_next_iter
        edge_vectors2.append(vector2)
        res = edge_vectors[0].view([self.batch_size, self.n_types])
        #res2 = edge_vectors2[0].view([self.batch_size, (self.n_types+self.n_types)*(self.n_types+self.n_types)])
        res2 = edge_vectors2[0].view([self.batch_size, (self.n_types+self.n_types)])
        return res,res2

    @staticmethod
    def train_step(model, optimizer, batch):
        model.train()
        optimizer.zero_grad()
        model(batch)
        criterion = nn.CrossEntropyLoss()
        loss = torch.mean(criterion(model.scores,model.labels))
        loss.backward()
        optimizer.step()

        return loss.item()

    @staticmethod
    def test_step(model,batch):
        model.eval()
        with torch.no_grad():
            model(batch)
            acc = (model.labels == model.scores.argmax(dim=1)).float().tolist()
            y_true = model.labels.tolist()
            y_pred = model.scores.argmax(dim=1).tolist()
            emb = model.embedding.tolist()

        return acc, model.scores_normalized.tolist(),y_true,y_pred,emb

    
    
    




