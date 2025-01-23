import copy

import numpy as np
from torch import nn
from torch.utils.data import Dataset, DataLoader
import torch
from torch.nn.functional import cosine_similarity
from sgd_clip import SGDClipGrad

class DatasetSplit(Dataset):
    def __init__(self, dataset, idxs):
        self.dataset = dataset
        self.idxs = list(idxs)

    def __len__(self):
        return len(self.idxs)

    def __getitem__(self, item):
        image, label = self.dataset[self.idxs[item]]
        return image, label


class LocalUpdate(object):
    def __init__(self, args, dataset, idxs, users_speed, train_data_num):
        self.args = args
        self.train_data_num = int(train_data_num * len(idxs))
        # self.logs = logs
        self.trainloader, self.testloader = self.train_val_test(dataset, list(idxs), users_speed)
        self.device = 'cuda' if (torch.cuda.is_available() and self.args.gpu != -1) else 'cpu'
        self.loss_func = nn.CrossEntropyLoss().to(self.device)

    def train_val_test(self, dataset, idxs, users_speed):
        # idxs_train = idxs[:int(0.8 * len(idxs))]

        idxs_train = np.random.choice(idxs, self.train_data_num, replace=False)

        # idxs_train = np.random.choice(idxs, int(0.4 * len(idxs)), replace=False)
        # idxs_test = idxs[int(0.8 * len(idxs)):]
        idxs_test = list(set(idxs) - set(idxs_train))
        # data = DatasetSplit(dataset, idxs_train)
        if users_speed == 1:  # fast_users:  batch:96
            trainloader = DataLoader(DatasetSplit(dataset, idxs_train),
                                     batch_size=self.args.fast_batch, shuffle=True)
        elif users_speed == 2:  # common_users:  batch:24
            trainloader = DataLoader(DatasetSplit(dataset, idxs_train),
                                     batch_size=self.args.common_batch, shuffle=True)
        else:  # slow_users:  batch:8
            trainloader = DataLoader(DatasetSplit(dataset, idxs_train),
                                     batch_size=self.args.slow_batch, shuffle=True)
        testloader = DataLoader(DatasetSplit(dataset, idxs_test),
                                batch_size=int(len(idxs_test) / 10), shuffle=False)
        return trainloader, testloader

    def update_weights(self, model, local_epoch):
        epoch_loss = []
        model.train()
        optimizer = torch.optim.SGD(model.parameters(), lr=self.args.lr, momentum=self.args.momentum)
        for lepoch in range(local_epoch):
            batch_loss = []
            for batch_idx, (images, labels) in enumerate(self.trainloader):
                images, labels = images.to(self.device), labels.to(self.device)
                out_put = model(images)
                loss = self.loss_func(out_put, labels)  # 一个batchsize的平均损失

                optimizer.zero_grad()
                loss.backward()
                optimizer.step()

                # self.logs.add_scalar("loss", loss.item())
                loss = loss.item()
                batch_loss.append(loss)

            epoch_loss.append(sum(batch_loss) / len(batch_loss))
            # test = sum(epoch_loss) / len(epoch_loss)
        # print("1")

        return model.state_dict(), sum(epoch_loss) / len(epoch_loss),  self.train_data_num  # 返回一个客户端的经过10次本地训练后的本地模型参数和平均训练损失

    def episode_update_weights(self, model, local_epoch):
        epoch_loss = []
        model.train()
        optimizer = SGDClipGrad(params=model.parameters(), lr=self.args.lr, momentum=self.args.momentum,
                                weight_decay=0, nesterov=False,
                                clipping_param=1.0)
        for lepoch in range(local_epoch):
            batch_loss = []
            for batch_idx, (images, labels) in enumerate(self.trainloader):
                images, labels = images.to(self.device), labels.to(self.device)
                out_put = model(images)
                loss = self.loss_func(out_put, labels)  # 一个batchsize的平均损失

                optimizer.zero_grad()
                loss.backward()
                optimizer.step(0.0)

                # self.logs.add_scalar("loss", loss.item())
                loss = loss.item()
                batch_loss.append(loss)

            epoch_loss.append(sum(batch_loss) / len(batch_loss))
            # test = sum(epoch_loss) / len(epoch_loss)
        # print("1")

        return model.state_dict(), sum(epoch_loss) / len(epoch_loss),  self.train_data_num  # 返回一个客户端的经过10次本地训练后的本地模型参数和平均训练损失
    def test_over_user(self, model):
        loss, total, acc = 0.0, 0.0, 0.0
        batch_loss = []
        model.eval()
        for batch_idx, (images, labels) in enumerate(self.testloader):
            images, labels = images.to(self.device), labels.to(self.device)
            output = model(images)  # 64*10: 64张图片 每张图片有10个分类的值
            loss = self.loss_func(output, labels)
            batch_loss.append(loss.item())

            # *************************************
            # 取output中每行最大值的idx(对应分类)， _表示最大的值  一共10个(10张图片有10个预测值)  tensor([*,*,*,*,...,*])
            _, pred_labels = torch.max(output, 1)
            pred_labels = pred_labels.view(-1)
            acc += torch.sum(torch.eq(pred_labels, labels)).item()
            # print("\ntest_over_user acc:{}".format(acc))
            total += len(labels)

        accuracy = acc / total
        return accuracy, sum(batch_loss) / len(batch_loss)


def global_agg_cos(local_w, global_w):
    cosine_sum = 0
    cosine = [torch.tensor(1.0) for i in range(len(local_w))]
    for idx in range(len(local_w)):
        local_w_v = []
        global_w_v = []
        for k, v in local_w[idx].items():
            local_w_v.append(v)
        for k, v in global_w.items():
            global_w_v.append(v)
        for i in range(len(local_w_v)):
            cos_layer = cosine_similarity(local_w_v[i].to(torch.float), global_w_v[i].to(torch.float), dim=-1).cpu()
            cos_layer = list(np.array(cos_layer).flatten())
            cosine[idx] += (sum(cos_layer) / len(cos_layer))  # 第i层所有权重相似度的平均
        cosine[idx] = cosine[idx] / len(local_w_v)  # 所有层的平均
        cosine_sum += cosine[idx]

    g_w = {}
    for key in global_w.keys():
        g_w[key] = 0
    for key in g_w.keys():
        for i in range(len(local_w)):
            g_w[key] += torch.mul(local_w[i][key], cosine[i] / cosine_sum)
        # g_w[key] = torch.div(g_w[key], len(local_w))
    return g_w


def global_agg(local_w, global_w, datanumfrc):
    # g_w = copy.deepcopy(global_w)
    # for key in g_w.keys():
    #     for i in range(len(local_w)):
    #         g_w[key] += local_w[i][key]
    #     g_w[key] = torch.div(g_w[key], len(local_w)+1)
    #####################################################
    g_w = copy.deepcopy(global_w)
    data_sum = sum(datanumfrc)
    local_num = len(datanumfrc)
    for key in local_w[0].keys():
        g_w[key] = torch.mul(g_w[key], 1.0/(local_num+1))
    for key in g_w.keys():
        for i in range(len(local_w)):
            rate_weight = datanumfrc[i]*local_num / (data_sum*(local_num+1))
            g_w[key] += torch.mul(local_w[i][key], rate_weight)
    ##########################
    # for key in g_w.keys():
    #     g_w[key] += global_w[key]
    #     g_w[key] = torch.div(g_w[key], 2)
    return g_w


def global_agg_hsafl(local_w, datanumfrc, local_ep):
    g_w = {}

    fen_mu = 0.0
    p = []
    for i in range(len(local_w)):
        fen_mu += datanumfrc[i]*local_ep[i]

    for i in range(len(local_w)):
        p.append((datanumfrc[i]*local_ep[i]) / fen_mu)
    for key in local_w[0].keys():
        g_w[key] = 0
    for key in g_w.keys():
        for i in range(len(local_w)):
            g_w[key] += torch.mul(local_w[i][key], p[i])
    return g_w

#
def test(args, model, dataset, idx):
    model.eval()
    batch_loss = []
    loss, total, acc = 0.0, 0.0, 0.0
    TestData = DataLoader(dataset=DatasetSplit(dataset, idx), batch_size=68, shuffle=False, )
    device = 'cuda' if (torch.cuda.is_available() and args.gpu != -1) else 'cpu'
    loss_func = nn.CrossEntropyLoss().to(device)
    for batch_idx, (images, labels) in enumerate(TestData):
        images, labels = images.to(device), labels.to(device)
        output = model(images)
        loss = loss_func(output, labels)
        batch_loss.append(loss.item())

        # *************************************
        _, pred_labels = torch.max(output, 1)
        pred_labels = pred_labels.view(-1)
        acc += torch.sum(torch.eq(pred_labels, labels)).item()
        total += len(labels)

    accuracy = acc / total

    return accuracy, sum(batch_loss) / len(batch_loss)

