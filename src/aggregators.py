import torch
import torch.nn as nn
from abc import abstractmethod
import torch.nn.functional as F


class Aggregator(nn.Module):
    def __init__(self, batch_size, input_dim, output_dim, act, self_included):
        super(Aggregator, self).__init__()
        self.batch_size = batch_size
        self.input_dim = input_dim
        self.output_dim = output_dim
        self.act = act
        self.act2 = F.relu
        self.self_included = self_included

    def forward(self, self_vectors, neighbor_vectors, masks,node_emb):
        # self_vectors: [batch_size, -1, input_dim]
        # neighbor_vectors: [batch_size, -1, 2, n_neighbor, input_dim]
        # masks: [batch_size, -1, 2, n_neighbor, 1]
        
        hyperedge_vectors = torch.mean(neighbor_vectors * masks, dim=-2)  # [batch_size, -1, 2, input_dim]
        hyperedge_vectors = torch.squeeze(hyperedge_vectors,1)
        hyperedge_v = torch.cat([hyperedge_vectors, node_emb],dim=2)
        hyperedge_v = torch.unsqueeze(hyperedge_v,1)
        outputs = self._call(self_vectors, hyperedge_v,node_emb)
        return outputs

    @abstractmethod
    def _call(self, self_vectors, hyperedge_vectors,node_emb):
        # self_vectors: [batch_size, -1, input_dim]
        # entity_vectors: [batch_size, -1, 2, input_dim]
        pass


class MeanAggregator(Aggregator):
    def __init__(self, batch_size, input_dim, output_dim, act=lambda x: x, self_included=True):
        super(MeanAggregator, self).__init__(batch_size, input_dim, output_dim, act, self_included)
        self.layer1 = nn.Linear((self.input_dim+self.input_dim), (self.input_dim+self.input_dim))
        #self.layer1 = nn.Linear((self.input_dim+self.input_dim+14), 64)
        self.layer2 = nn.Linear((self.input_dim+self.input_dim), self.output_dim)
        nn.init.xavier_uniform_(self.layer1.weight)
        nn.init.xavier_uniform_(self.layer2.weight)

    def _call(self, self_vectors, hyperedge_vectors,node_emb):
        self.act2 = F.relu
        maxout = torch.amax(hyperedge_vectors,dim=-2)
        minout = torch.amin(hyperedge_vectors,dim=-2)
        output = torch.sub(maxout,minout)
        output1 = self.layer1(output)  # [-1, output_dim]
        output2 = self.layer2(output1)

        output1 = output1.view([self.batch_size, -1, (self.input_dim+self.input_dim)])  # [batch_size, -1, output_dim]

        return self.act(output2), self.act2(output1)


class ConcatAggregator(Aggregator):
    def __init__(self, batch_size, input_dim, output_dim, act=lambda x: x, self_included=True):
        super(ConcatAggregator, self).__init__(batch_size, input_dim, output_dim, act, self_included)

        multiplier = 5 if self_included else 6

        self.layer = nn.Linear(self.input_dim * multiplier, self.output_dim)
        nn.init.xavier_uniform_(self.layer.weight)

    def _call(self, self_vectors, entity_vectors):
        # self_vectors: [batch_size, -1, input_dim]
        # entity_vectors: [batch_size, -1, 2, input_dim]

        output = entity_vectors.view([-1, self.input_dim * 5])  # [-1, input_dim * 2]
        if self.self_included:
            self_vectors = self_vectors.view([-1, self.input_dim])  # [-1, input_dim]
            output = torch.cat([self_vectors, output], dim=-1)  # [-1, input_dim * 3]
        output = self.layer(output)  # [-1, output_dim]
        output = output.view([self.batch_size, -1, self.output_dim])  # [batch_size, -1, output_dim]

        return self.act(output)

