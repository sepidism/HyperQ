import argparse
from data_loader import load_data
from train import train


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--cuda', default=False,help='use gpu',action='store_true')

    #========== Pubmed ==========#
    parser.add_argument('--dataset', type=str, default='fb',help='dataset name')
    parser.add_argument('--epoch', type=int, default=8,help='number of epochs')
    parser.add_argument('--hedge_size', type=int, default=6,help='hyperedge sample size')
    parser.add_argument('--batch_size', type=int, default=1, help='batch size')
    parser.add_argument('--dim', type=int, default=64, help='hidden dimension')
    parser.add_argument('--l2', type=float, default=1e-7, help='l2 regularization weight')
    parser.add_argument('--lr', type=float, default=5e-3, help='learning rate')
    parser.add_argument('--feature_type', type=str, default='id', help='type of relation features: id, bow, bert')
    parser.add_argument('--neighbor_samples', type=int, default=6, help='number of sampled neighbors')
    parser.add_argument('--knowledge', type=bool, default=True, help='Is this a knowledge hypergraph?')

    args = parser.parse_args()
    data = load_data(args)
    train(args,data)
    

if __name__ == '__main__':
    main()
    

