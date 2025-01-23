import argparse

def args_parser():
    parser = argparse.ArgumentParser()

    parser.add_argument('--iid', type=int, default=0,
                        help='Default set to IID. Set to 0 for non-IID.')
    parser.add_argument('--dataset', type=str, default='emnist', help="name \
                        of dataset: FMNIST----fashion mnist")
    parser.add_argument('--user_num', type=int, default=100,
                        help="number of users: K")
    parser.add_argument('--epochs', type=int, default=10,
                        help="number of rounds of training")
    parser.add_argument('--frac', type=float, default=0.1,
                        help='the fraction of clients: C')
    parser.add_argument('--local_ep', type=int, default=10,
                        help="the number of init local epochs: E")
    parser.add_argument('--lr', type=float, default=0.01,
                        help='learning rate')
    parser.add_argument('--momentum', type=float, default=0.5,
                        help='SGD momentum (default: 0.5)')
    parser.add_argument('--gpu', type=int, default=0,
                        help="GPU ID, -1 for CPU")
    parser.add_argument('--model', type=str, default='CNN', help='model name')

    parser.add_argument("--slow_batch", type=int, default=64,
                        help="slow client local batch size")
    parser.add_argument('--common_batch', type=int, default=64,
                        help="common client local batch size")
    parser.add_argument('--fast_batch', type=int, default=64,
                        help="fast client local batch size")
    parser.add_argument('--common_frac', type=float, default=0.5,
                        help='the fraction of common clients: C')
    parser.add_argument('--fastOrslow_frac', type=float, default=0.25,
                        help='the fraction of fastOrslow clients: C')


    args = parser.parse_args()

    return args