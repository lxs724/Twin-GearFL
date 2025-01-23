# Dataset 获取数据和label  Dataloder: 对获取的数据打包
import numpy as np
import torchvision.datasets
from local_update import *
from torch.utils.tensorboard import SummaryWriter
from torchvision import transforms
from torch.utils.data import Dataset, DataLoader


class DatasetSplit(Dataset):
    def __init__(self, dataset, idxs):
        self.dataset = dataset
        self.idxs = list(idxs)

    def __len__(self):
        return len(self.idxs)

    def __getitem__(self, item):
        image, label = self.dataset[self.idxs[item]]
        return image, label


def CIFAR10_iid(dataset, user_num):
    num_items = int(len(dataset) / user_num)
    # num_items = int(len(dataset.idxs) / user_num)
    dict_users, all_idxs = {}, [i for i in range(len(dataset))]
    # dict_users = {}
    # all_idxs = dataset.idxs
    for i in range(user_num):
        dict_users[i] = set(np.random.choice(all_idxs, num_items, replace=False))
        all_idxs = list(set(all_idxs) - dict_users[i])
        dict_users[i] = list(dict_users[i])
    return dict_users


def SVHN_iid(dataset, user_num):
    num_items = int(len(dataset) / user_num)
    # num_items = int(len(dataset.idxs) / user_num)
    dict_users, all_idxs = {}, [i for i in range(len(dataset))]
    # dict_users = {}
    # all_idxs = dataset.idxs
    for i in range(user_num):
        dict_users[i] = set(np.random.choice(all_idxs, num_items, replace=False))
        all_idxs = list(set(all_idxs) - dict_users[i])
        dict_users[i] = list(dict_users[i])
    return dict_users


def CIFAR100_iid(dataset, user_num):
    num_items = int(len(dataset) / user_num)
    # num_items = int(len(dataset.idxs) / user_num)
    dict_users, all_idxs = {}, [i for i in range(len(dataset))]
    # dict_users = {}
    # all_idxs = dataset.idxs
    for i in range(user_num):
        dict_users[i] = set(np.random.choice(all_idxs, num_items, replace=False))
        all_idxs = list(set(all_idxs) - dict_users[i])
        dict_users[i] = list(dict_users[i])
    return dict_users


def MNIST_iid(dataset, user_num):
    num_items = int(len(dataset) / user_num)
    # num_items = int(len(dataset.idxs) / user_num)
    dict_users, all_idxs = {}, [i for i in range(len(dataset))]
    # dict_users = {}
    # all_idxs = dataset.idxs
    for i in range(user_num):
        dict_users[i] = set(np.random.choice(all_idxs, num_items, replace=False))
        all_idxs = list(set(all_idxs) - dict_users[i])
        dict_users[i] = list(dict_users[i])
    return dict_users


def CIFAR10_noniid(dataset, num_users):
    """
    Sample non-I.I.D client data from CIFAR10 dataset
    :param dataset:
    :param num_users:
    :return:
    """
    num_shards, num_imgs = 200, 250
    idx_shard = [i for i in range(num_shards)]
    dict_users = {i: [] for i in range(num_users)}
    idxs = np.arange(num_shards * num_imgs)  # 数据集的idxs 0-49999
    # labels = dataset.train_labels.numpy()
    labels = np.array(dataset.targets)  # 获取每个标签50000个

    # sort labels
    idxs_labels = np.vstack((idxs, labels))  # 将labels转为二维数组(2，50000)，第一行是0-49999，第二行是对应的label
    idxs_labels = idxs_labels[:, idxs_labels[1, :].argsort()]  # 按照label排序 第一行跟着label变
    idxs = idxs_labels[0, :]  # 获取排序后的idxs(第一行)

    # divide and assign
    for i in range(num_users):
        rand_set = set(np.random.choice(idx_shard, 2, replace=False))  # 200个shard中选2个
        idx_shard = list(set(idx_shard) - rand_set)  # 把选出去的从shard中删掉
        for rand in rand_set:
            test = idxs[rand * num_imgs:(rand + 1) * num_imgs]  # 获取indx：从rand*250后连续选250个
            dict_users[i].extend(test)  # 列表拼接
            # print(dict_users[i], type(dict_users[i][0]), len(dict_users[i]))
            # print("1")
            # dict_users[i] = np.concatenate(
            #     (dict_users[i], test.astype(np.int)), axis=0)
    return dict_users


def SVHN_noniid(dataset, num_users):
    """
    Sample non-I.I.D client data from SVHN dataset
    :param dataset:
    :param num_users:
    :return:
    """
    num_shards, num_imgs = 244, 300
    idx_shard = [i for i in range(num_shards)]
    dict_users = {i: [] for i in range(num_users)}
    idxs = np.arange(num_shards * num_imgs)  # 数据集的idxs 0-49999
    # labels = dataset.train_labels.numpy()
    labels = np.array(dataset.labels)  # 获取每个标签50000个

    # sort labels
    idxs_labels = np.vstack((idxs, labels[:73200]))  # 将labels转为二维数组(2，73200)，第一行是0-73199，第二行是对应的label
    idxs_labels = idxs_labels[:, idxs_labels[1, :].argsort()]  # 按照label排序 第一行跟着label变
    idxs = idxs_labels[0, :]  # 获取排序后的idxs(第一行)

    # divide and assign
    for i in range(num_users):
        rand_set = set(np.random.choice(idx_shard, 2, replace=False))  # 200个shard中选2个
        idx_shard = list(set(idx_shard) - rand_set)  # 把选出去的从shard中删掉
        for rand in rand_set:
            test = idxs[rand * num_imgs:(rand + 1) * num_imgs]  # 获取indx：从rand*250后连续选250个
            dict_users[i].extend(test)  # 列表拼接
            # print(dict_users[i], type(dict_users[i][0]), len(dict_users[i]))
            # print("1")
            # dict_users[i] = np.concatenate(
            #     (dict_users[i], test.astype(np.int)), axis=0)
    return dict_users


def CIFAR100_noniid(dataset, num_users):
    """
    Sample non-I.I.D client data from CIFAR10 dataset
    :param dataset:
    :param num_users:
    :return:
    """
    num_shards, num_imgs = 200, 250
    idx_shard = [i for i in range(num_shards)]
    dict_users = {i: [] for i in range(num_users)}
    idxs = np.arange(num_shards * num_imgs)  # 数据集的idxs 0-49999
    # labels = dataset.train_labels.numpy()
    labels = np.array(dataset.targets)  # 获取每个标签50000个

    # sort labels
    idxs_labels = np.vstack((idxs, labels))  # 将labels转为二维数组(2，50000)，第一行是0-49999，第二行是对应的label
    idxs_labels = idxs_labels[:, idxs_labels[1, :].argsort()]  # 按照label排序 第一行跟着label变
    idxs = idxs_labels[0, :]  # 获取排序后的idxs(第一行)

    # divide and assign
    for i in range(num_users):
        rand_set = set(np.random.choice(idx_shard, 2, replace=False))  # 200个shard中选2个
        idx_shard = list(set(idx_shard) - rand_set)  # 把选出去的从shard中删掉
        for rand in rand_set:
            test = idxs[rand * num_imgs:(rand + 1) * num_imgs]  # 获取indx：从rand*250后连续选250个
            dict_users[i].extend(test)  # 列表拼接
            # print(dict_users[i], type(dict_users[i][0]), len(dict_users[i]))
            # print("1")
            # dict_users[i] = np.concatenate(
            #     (dict_users[i], test.astype(np.int)), axis=0)
    return dict_users


def MNIST_noniid(dataset, num_users):
    """
    Sample non-I.I.D client data from CIFAR10 dataset
    :param dataset:
    :param num_users:
    :return:
    """
    num_shards, num_imgs = 200, 300
    idx_shard = [i for i in range(num_shards)]
    dict_users = {i: [] for i in range(num_users)}
    idxs = np.arange(num_shards * num_imgs)  # 数据集的idxs 0-49999
    # labels = dataset.train_labels.numpy()
    labels = np.array(dataset.targets)  # 获取每个标签50000个

    # sort labels
    idxs_labels = np.vstack((idxs, labels))  # 将labels转为二维数组(2，50000)，第一行是0-49999，第二行是对应的label
    idxs_labels = idxs_labels[:, idxs_labels[1, :].argsort()]  # 按照label排序 第一行跟着label变
    idxs = idxs_labels[0, :]  # 获取排序后的idxs(第一行)

    # divide and assign
    for i in range(num_users):
        rand_set = set(np.random.choice(idx_shard, 2, replace=False))  # 200个shard中选2个
        idx_shard = list(set(idx_shard) - rand_set)  # 把选出去的从shard中删掉
        for rand in rand_set:
            test = idxs[rand * num_imgs:(rand + 1) * num_imgs]  # 获取indx：从rand*250后连续选250个
            dict_users[i].extend(test)  # 列表拼接
            # print(dict_users[i], type(dict_users[i][0]), len(dict_users[i]))
            # print("1")
            # dict_users[i] = np.concatenate(
            #     (dict_users[i], test.astype(np.int)), axis=0)
    return dict_users


def getData(args):
    if args.dataset == "CIFAR10":
        transform_apply = transforms.Compose(
            [transforms.ToTensor(), transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))])
        train_data = torchvision.datasets.CIFAR10(root="../data/CIFAR10", train=True, transform=transform_apply)
        test_data = torchvision.datasets.CIFAR10(root="../data/CIFAR10", train=False, transform=transform_apply)

        # 将50000张训练图片随机分为40000张训练 和 10000张测试，并分发给100个客户端
        # indexs = [i for i in range(len(train_data))]
        # train_idxs = list(np.random.choice(indexs, int(len(train_data) * 0.8), replace=False))
        # test_idxs = list(set(indexs) - set(train_idxs))
        # users_train_data = DatasetSplit(train_data, train_idxs)
        # users_test_data = DatasetSplit(train_data, test_idxs)
        # print(len(users_train_data))
        # print(users_train_data.dataset.targets)

        if args.iid:
            # user_groups = CIFAR10_iid(users_train_data, args.user_num)
            user_groups = CIFAR10_iid(train_data, args.user_num)
        else:
            user_groups = CIFAR10_noniid(train_data, args.user_num)
            # user_groups_test = CIFAR10_iid(users_test_data, args.user_num)

    elif args.dataset == "MNIST":
        transform_apply = transforms.Compose(
            [transforms.ToTensor(), transforms.Normalize((0.1307,), (0.3081,))])
        train_data = torchvision.datasets.MNIST(root="../data/mnist", train=True, transform=transform_apply)
        test_data = torchvision.datasets.MNIST(root="../data/mnist", train=False, transform=transform_apply)
        if args.iid:
            user_groups = MNIST_iid(train_data, args.user_num)
        else:
            user_groups = MNIST_noniid(train_data, args.user_num)

    elif args.dataset == "FMNIST":
        transform_apply = transforms.Compose(
            [transforms.ToTensor(), transforms.Normalize((0.1307,), (0.3081,))])
        train_data = torchvision.datasets.FashionMNIST(root="../data/fashion_mnist", train=True,
                                                       transform=transform_apply)
        test_data = torchvision.datasets.FashionMNIST(root="../data/fashion_mnist", train=False,
                                                      transform=transform_apply)
        if args.iid:
            user_groups = MNIST_iid(train_data, args.user_num)
        else:
            user_groups = MNIST_noniid(train_data, args.user_num)

    elif args.dataset == "CIFAR100":
        transform_apply = transforms.Compose(
            [transforms.ToTensor(), transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))])
        train_data = torchvision.datasets.CIFAR100(root="../data/CIFAR100", train=True, transform=transform_apply,
                                                   download=True)
        test_data = torchvision.datasets.CIFAR100(root="../data/CIFAR100", train=False, transform=transform_apply,
                                                  download=True)
        if args.iid:
            # user_groups = CIFAR10_iid(users_train_data, args.user_num)
            user_groups = CIFAR100_iid(train_data, args.user_num)
        else:
            user_groups = CIFAR100_noniid(train_data, args.user_num)

    elif args.dataset == "SVHN":
        transform_apply = transforms.Compose(
            [transforms.ToTensor(), transforms.Normalize(mean=[0.485, 0.456, 0.106], std=[0.229, 0.224, 0.225])])
        train_data = torchvision.datasets.SVHN(
            root="../data/SVHN",
            split='train',
            transform=transform_apply,
            download=False,
        )
        # EMNIST 手写字母 测试集
        test_data = torchvision.datasets.SVHN(
            root="../data/SVHN",
            split='test',
            transform=transform_apply,
            download=False,
        )
        if args.iid:
            # user_groups = CIFAR10_iid(users_train_data, args.user_num)
            user_groups = SVHN_iid(train_data, args.user_num)
        else:
            user_groups = SVHN_noniid(train_data, args.user_num)


    groups_train = {i: [] for i in range(args.user_num)}
    idxs_test = {i: [] for i in range(args.user_num)}
    for i in range(args.user_num):
        groups_train[i] = np.random.choice(user_groups[i], int(0.8 * len(user_groups[i])), replace=False)
        idxs_test[i] = list(set(user_groups[i]) - set(groups_train[i]))

    return train_data, idxs_test, groups_train
