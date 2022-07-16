import argparse
from data_loader import load_data
from train import train


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--cuda', default=False,help='use gpu',action='store_true')

    #========== Pubmed ==========#
    parser.add_argument('--dataset', type=str, default='gps',help='dataset name')
    parser.add_argument('--epoch', type=int, default=3,help='number of epochs')
    parser.add_argument('--hedge_size', type=int, default=12,help='hyperedge sample size')
    parser.add_argument('--batch_size', type=int, default=1, help='batch size')
    parser.add_argument('--dim', type=int, default=64, help='hidden dimension')
    parser.add_argument('--l2', type=float, default=1e-7, help='l2 regularization weight')
    parser.add_argument('--lr', type=float, default=5e-3, help='learning rate')
    parser.add_argument('--feature_type', type=str, default='id', help='type of relation features: id, bow, bert')
    parser.add_argument('--neighbor_samples', type=int, default=12, help='number of sampled neighbors')
    parser.add_argument('--context_hops', type=int, default=1, help='number of context hops')
    parser.add_argument('--prediction', type=bool, default=False, help='Is this a prediction task?')

    args = parser.parse_args()
    data = load_data(args)
    train(args,data)
    

if __name__ == '__main__':
    main()
    

