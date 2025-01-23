import copy
import pickle
import time

import numpy as np
from client_selection import *

from torch.utils.tensorboard import SummaryWriter
from options import args_parser
from my_data import *
from my_model import *
from local_update import *
from set_local_epoch import *
from tqdm import tqdm
import matplotlib
import matplotlib.pyplot as plt

if __name__ == '__main__':

    matplotlib.use('Agg')

    writer = SummaryWriter("logs")

    args = args_parser()

    train_data, test_data_idx, user_groups = getData(args)
    img_size = train_data[0][0].shape
    # train_dataloader = DataLoader(train_data, batch_size=args.local_bs)
    # test_dataloader = DataLoader(test_data,batch_size=args.local_bs)
    cluster_model = {}
    cluster_weights = {}
    device = 'cuda' if (torch.cuda.is_available() and args.gpu != -1) else 'cpu'
    if args.dataset == "CIFAR10":
        for i in range(4):
            cluster_model[i] = CIFAR10_model()
        global_model = CIFAR10_model()
        local_model = CIFAR10_model()
    elif args.dataset == "MNIST":
        args.lr = 0.01
        if args.model == "CNN":
            for i in range(2):
                cluster_model[i] = mnist_cnn()
            global_model = mnist_cnn()
            local_model = mnist_cnn()
        elif args.model == "MLP":
            len_in = 1
            for x in img_size:
                len_in *= x
            for i in range(2):
                cluster_model[i] = MLP(dim_in=len_in, dim_hidden=200, dim_out=10)
            global_model = MLP(dim_in=len_in, dim_hidden=200, dim_out=10)
            local_model = MLP(dim_in=len_in, dim_hidden=200, dim_out=10)
        else:
            print("参数错误！")
            exit(0)
    elif args.dataset == "FMNIST":
        args.lr = 0.01
        for i in range(4):
            cluster_model[i] = CNNFashion_Mnist()
        global_model = CNNFashion_Mnist()
        local_model = CNNFashion_Mnist()
    elif args.dataset == "SVHN":
        args.lr = 0.01
        for i in range(4):
            cluster_model[i] = SVHN_model(Basicblock, [1, 1, 1, 1], 10)
        global_model = SVHN_model(Basicblock, [1, 1, 1, 1], 10)
        local_model = SVHN_model(Basicblock, [1, 1, 1, 1], 10)
    else:
        print("参数错误！")
        exit(0)
    global_model.to(device)
    local_model.to(device)
    for i in range(len(cluster_model)):
        cluster_model[i].to(device)
        cluster_model[i].train()
        cluster_weights[i] = cluster_model[i].state_dict()
    global_model.train()
    print(global_model)

    epoch_train_time = [0.0 for i in range(args.epochs)]
    total_train_time = [0.0 for i in range(args.epochs)]

    eopch_local_ep = [0.0 for i in range(args.epochs)]
    total_local_ep = [0.0 for i in range(args.epochs)]      # 用本地迭代次数来表示资源消耗

    global_weights = global_model.state_dict()


    users_cluster = [100 for i in range(args.user_num)]     # 初始化客户端集群身份
    train_phase_loss, train_test_acc, train_test_loss = [], [], []
    testdataset_acc, testdataset_loss = [], []
    test_phase_acc = 0.0
    test_user_dict_acc = {i: [] for i in range(args.user_num)}
    test_user_dict_loss = {i: [] for i in range(args.user_num)}

    user_speed = []  # 用户训练速度：1表示快速，2表示正常，3表示慢速
    perEpochTime = []
    for idx in range(int(args.user_num * args.fastOrslow_frac)):
        user_speed.append(1)
        perEpochTime.append(1.0)
    for idx in range(int(args.user_num * args.common_frac)):
        user_speed.append(2)
        perEpochTime.append(2.1)
    for idx in range(int(args.user_num * args.fastOrslow_frac)):
        user_speed.append(3)
        perEpochTime.append(3.2)

    users_train_data_num = []
    for idx in range(args.user_num):
        users_train_data_num.append(0.9)
        users_train_data_num.append(0.9)
        users_train_data_num.append(0.9)
        users_train_data_num.append(0.9)

    users_training_time = [0.0 for i in range(args.user_num)]
    users_training_loss = [0.0 for i in range(args.user_num)]
    users_local_ep = [args.local_ep for i in range(args.user_num)]
    train_time = [0.0 for i in range(args.epochs)]      # 记录没轮中最慢客户端的训练时间
    dev_user_time, dev_user_loss, users_level = {}, {}, {}
    k = [0 for i in range(args.user_num)]  # k为一个list，表示每个客户端最近一次参与训练的全局轮数

    z = [0 for i in range(args.user_num)]  # z为一个list，表示每个客户端参与训练的次数

    # local_weights_all = []
    # constraint = []
    # Constraint_threshold = 100
    # for u in range(args.user_num):
    #     local_weights_all.append(copy.deepcopy(global_weights))
    #     constraint.append(Constraint_threshold)
    m = max(int(args.frac * args.user_num), 1)
    idxs_users = np.random.choice(range(args.user_num), m, replace=False)
    # zzz = np.append(users_training_time, idxs_users)
    # 提前训练一次，解决时间bug
    for idxs in range(1):
        local = LocalUpdate(args=args, dataset=train_data, idxs=user_groups[idxs], users_speed=user_speed[idxs], train_data_num=users_train_data_num[idxs])
        w, loss, _ = local.update_weights(model=copy.deepcopy(global_model), local_epoch=users_local_ep[idxs])
    start_time = time.time()
    for i in tqdm(range(args.epochs)):
        perRoundClusterClientNum = [0 for _ in range(len(cluster_model))]    # 每轮选择的客户端中 每个集群的客户端数量

        local_weights, local_loss = [[] for _ in range(len(cluster_model))], []     # 本地权重==【[],[],[],[]】 每个[]表示集群身份对应的本地模型
        localepo_round = []
        datanumfrc = [[] for _ in range(len(cluster_model))]
        print(f'\n | Global Training Round : {i + 1} |\n')

        # 估计每个客户端的集群身份
        for x in range(args.user_num):
            cluster_loss = []
            local = LocalUpdate(args=args, dataset=train_data, idxs=user_groups[x], users_speed=user_speed[x],
                                train_data_num=users_train_data_num[x])
            for ii in range(len(cluster_model)):
                acc, loss = local.test_over_user(cluster_model[ii])
                cluster_loss.append(loss)
            users_cluster[x] = cluster_loss.index(min(cluster_loss))      # 找到损失最低的索引 即获得每个客户端的集群身份
        print(users_cluster)
        # 开始训练
        for idxs in idxs_users:
            pi = users_cluster[idxs]
            perRoundClusterClientNum[pi] += 1

            k[idxs] = i + 1
            z[idxs] += 1
            local_model = copy.deepcopy(cluster_model[pi])

            start_time_users = time.time()
            # constraint[idxs] -= 1  # 对选择的客户端的约束因子减一
            local = LocalUpdate(args=args, dataset=train_data, idxs=user_groups[idxs], users_speed=user_speed[idxs], train_data_num=users_train_data_num[idxs])

            w, loss, datanum = local.update_weights(model=local_model, local_epoch=users_local_ep[idxs])
            # 计算本地模型精度F
            local_model.load_state_dict(w)
            _, users_training_loss[idxs] = local.test_over_user(local_model)
            # print("第{}个用户的(验证集)训练损失为{}".format(idxs, users_training_loss[idxs]))
            # loss为第idxs个客户端的第i轮训练的平均训练损失
            # local_weights_all[idxs] = copy.deepcopy(w)
            # 在local_weights列表中添加字典w  local_weights标号表示第几个客户端，字典中key为模型层名称，value为对应的权重值
            local_weights[pi].append(copy.deepcopy(w))
            datanumfrc[pi].append(datanum)
            local_loss.append(copy.deepcopy(loss))  # local_loss是参与训练的所有客户端的训练损失

            end_time_users = time.time()
            users_training_time[idxs] = perEpochTime[idxs]*users_local_ep[idxs]  # 每个客户端训练花费的时间  调整迭代次数

            epoch_train_time[i] += users_training_time[idxs]
            eopch_local_ep[i] += users_local_ep[idxs]

            localepo_round.append(users_local_ep[idxs])

        print("本地迭代次数为：{}".format(localepo_round))
        print("每个集群客户端个数为：{}".format(perRoundClusterClientNum))

        for idxs in idxs_users:
            if train_time[i] < users_training_time[idxs]:
                train_time[i] = users_training_time[idxs]
        if i == 0:
            total_train_time[i] = epoch_train_time[i]
            total_local_ep[i] = eopch_local_ep[i]
        else:
            total_train_time[i] = total_train_time[i-1] + epoch_train_time[i]
            total_local_ep[i] = total_local_ep[i-1] + eopch_local_ep[i]
        print("第{}轮的本地模型的训练损失：{}".format(i + 1, local_loss))
        print("本轮训练时间(最慢客户端)为：{}".format(train_time[i]))
        loss_avg = sum(local_loss) / len(local_loss)  # 计算idxs_users个客户端的平均训练损失
        # print("loss_avg_train = {}".format(loss_avg))
        train_phase_loss.append(loss_avg)

        # global_weights = global_agg_cos(local_w=local_weights, global_w=global_weights)  # 更新全局权重

        # 聚合每一个集群模型，如果本轮集群内没有客户端被选中，则不变
        for p in range(len(cluster_model)):
            if perRoundClusterClientNum[p] == 0:
                cluster_model[p].load_state_dict(cluster_weights[p])
            else:
                cluster_weights[p] = global_agg(local_w=local_weights[p], global_w=cluster_weights[p], datanumfrc=datanumfrc[p])
                cluster_model[p].load_state_dict(cluster_weights[p])
        # global_weights = global_agg_hsafl(local_w=local_weights, datanumfrc=datanumfrc, local_ep=localepo_round)

        global_model.load_state_dict(global_weights)  # 将模型权重加载到模型结构中

        if i != args.epochs-1:
            # 获取客户端水平， 本地迭代次数， 客户端的时间差异， 客户端的精度差异
            # m = 0
            # del_idxs = []
            users_level, users_local_ep, dev_user_time, dev_user_loss = set_local_epoch(idxs_users=idxs_users,
                                                                                        users_time=users_training_time,
                                                                                        users_loss=users_training_loss,
                                                                                        local_epoch=users_local_ep,
                                                                                        dev_user_time=dev_user_time,
                                                                                        dev_user_loss=dev_user_loss,
                                                                                        users_level=users_level)

            # for idxs in idxs_users:
            #     # print("第{}个客户端下次本地迭代次数为{}".format(idxs, users_local_ep[idxs]))
            #     # if users_level[idxs] == 4:
            #     del_idxs.append(idxs)
            #     m += 1      # 删除客户端的个数 并记录该客户端的idxs  （这里不能删除，因为下一行代码要用）
            # print(m)
            idxs_users = client_selection(idxs_users=idxs_users, k=k, dev_user_loss=dev_user_loss,
                                           dev_user_time=dev_user_time, args=args, m=m)
            # for idxs in del_idxs:
            #     idxs_users = np.delete(idxs_users, np.where(idxs_users == idxs))
            # idxs_users = np.append(idxs_users, idxs_result)   # 获得参与下一轮训练的客户端

        # 在用户训练数据集上的验证集上测试 第i轮通信的全局模型的精度和损失
        test_users_acc, test_users_loss = [], []
        for p in range(len(cluster_model)):
            cluster_model[p].eval()
        global_model.eval()  # 停止更新参数(不反向传播)
        with torch.no_grad():  # 不更新梯度
            for idxs in range(args.user_num):
                pi = users_cluster[idxs]

                local = LocalUpdate(args=args, dataset=train_data, idxs=user_groups[idxs], users_speed=user_speed[idxs], train_data_num=users_train_data_num[idxs])
                acc, loss = local.test_over_user(cluster_model[pi])  # loss为 第idxs个客户端 在 第i轮中 对聚合好的全局模型 测试的损失值
                test_users_acc.append(acc)
                test_users_loss.append(loss)  # 100个客户端的损失值

                test_user_dict_acc[idxs].append(acc)  # key: 客户端的index
                test_user_dict_loss[idxs].append(loss)  # value: 客户端每轮的精度或损失 list[]，长度为总通信轮数
                # if (i+1) % 5 == 0:  # 每5轮画一次图
                #     writer.add_scalar("第{}轮聚合后的所有客户端对全局模型的损失:x-use_idx".format(i + 1), loss, global_step=idxs)
                #     writer.add_scalar("第{}轮聚合后的所有客户端对全局模型的精度:x-use_idx".format(i + 1), acc,
                #                       global_step=idxs)
                #     print("\n第{}轮的聚合后第{}个客户端的精度为：{}，损失为：{}".format(i+1, idxs, acc, loss))
            train_test_acc.append(sum(test_users_acc) / len(test_users_acc))  # 每个通信回合中user其验证集在模型上的平均精度和损失 100个求平均
            train_test_loss.append(sum(test_users_loss) / len(test_users_loss))
            writer.add_scalar("全局模型的平均损失:x-epoch", train_test_loss[i], global_step=i)
        print("\n第{}轮的聚合后的全局模型对所有客户端的平均精度为：{}，平均损失为：{}".format(i + 1, train_test_acc[i], train_phase_loss[i]))

        # 测试阶段
        test_acc, test_loss = 0.0, 0.0
        for idxs in range(args.user_num):
            test_phase_acc, test_phase_loss = test(args, cluster_model[users_cluster[idxs]], train_data,
                                                   test_data_idx[idxs])
            test_acc += test_phase_acc
            test_loss += test_phase_loss

        test_acc /= args.user_num
        test_loss /= args.user_num
        print("\n测试数据集的精度为{}，损失为{}".format(test_acc, test_loss))
        testdataset_acc.append(test_acc)
        testdataset_loss.append(test_loss)

    writer.close()
    #
    file_name = '../save/data/fedSysHe3_alpha1.0_c4_{}_{}3_epo[{}]_C[{}]_iid[{}]_E[{}]_fastorslow[{}].pkl'. \
        format(args.dataset, args.model, args.epochs, args.frac, args.iid,
               args.local_ep, args.fastOrslow_frac)
    with open(file_name, 'wb') as f:
        pickle.dump([train_test_acc, train_phase_loss, test_user_dict_acc, test_user_dict_loss, testdataset_acc, train_time, total_train_time, total_local_ep], f)
        # 训练中所有用户的测试平均精度  训练阶段损失    用户测试集上的精度字典     用户测试集上的损失字典  测试集的精度  训练时间

    print('\n Total Run Time: {0:0.4f} s'.format(time.time() - start_time))
    #
    # plt.figure()
    # plt.title("Training Loss vs Communication rounds")
    # plt.plot(range(len(train_phase_loss)), train_phase_loss, color='r')
    # plt.ylabel('Training loss')
    # plt.xlabel('Communication Rounds')
    # plt.savefig("../save/figure/fedSysHe3_alpha1.0_c4_{}_{}3_{}_C[{}]_iid[{}]_E[{}]_fastorslow[{}]_loss.png".
    #             format(args.dataset, args.model, args.epochs, args.frac,
    #                    args.iid, args.local_ep, args.fastOrslow_frac))
    # plt.figure()
    # plt.title('Average Accuracy vs Communication rounds')
    # plt.plot(range(len(train_test_acc)), train_test_acc, color='k')
    # plt.ylabel('Average Accuracy')
    # plt.xlabel('Communication Rounds')
    # plt.savefig("../save/figure/fedSysHe3_alpha1.0_c4_{}_{}3_{}_C[{}]_iid[{}]_E[{}]_fastorslow[{}]_acc.png".
    #             format(args.dataset, args.model, args.epochs, args.frac,
    #                    args.iid, args.local_ep, args.fastOrslow_frac))
    # plt.figure()
    # plt.title('Test Accuracy vs Communication rounds')
    # plt.plot(range(len(testdataset_acc)), testdataset_acc, color='g')
    # plt.ylabel('Test Accuracy')
    # plt.xlabel('Communication Rounds')
    # plt.savefig("../save/figure/fedSysHe3_alpha1.0_c4_{}_{}3_{}_C[{}]_iid[{}]_E[{}]_fastorslow[{}]_test_acc.png".
    #             format(args.dataset, args.model, args.epochs, args.frac,
    #                    args.iid, args.local_ep, args.fastOrslow_frac))
    #
    # plt.figure()
    # plt.title('Test Accuracy vs time')
    # plt.plot(total_train_time, train_test_acc, color='y')
    # plt.ylabel('Test Accuracy')
    # plt.xlabel('time(s)')
    # plt.savefig("../save/figure/fedSysHe3_alpha1.0_c4_{}_{}3_{}_C[{}]_iid[{}]_E[{}]_fastorslow[{}]_test_acc_for_time.png".
    #             format(args.dataset, args.model, args.epochs, args.frac,
    #                    args.iid, args.local_ep, args.fastOrslow_frac))
    #
    # plt.figure()
    # plt.title('Test Accuracy vs total epoch number')
    # plt.plot(total_local_ep, train_test_acc, color='b')
    # plt.ylabel('Test Accuracy')
    # plt.xlabel('total epoch number')
    # plt.savefig("../save/figure/fedSysHe3_alpha1.0_c4_{}_{}3_{}_C[{}]_iid[{}]_E[{}]_fastorslow[{}]_test_acc_for_epo.png".
    #             format(args.dataset, args.model, args.epochs, args.frac,
    #                    args.iid, args.local_ep, args.fastOrslow_frac))
    print(k)
    print(z)
    print(users_level)
    print(users_training_loss)
    print(users_training_time)
    print(users_local_ep)
    print(train_time)
    print("时间：", sum(train_time))
    print(epoch_train_time)
    print(total_train_time)
    print(total_train_time[args.epochs-1])
    print(testdataset_acc[args.epochs-1])
    print(total_local_ep[args.epochs-1])
    print(train_test_acc[args.epochs-1])
    print(users_cluster)
    # train_size = len(train_dataloader)
    # test_size = len(test_data)
    # print("训练数据集的长度为{}，测试集的长度为{}".format(train_size, test_size))   # 字符串格式化format  将train_size数据转化为字符串格式
