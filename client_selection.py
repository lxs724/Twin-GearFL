import numpy as np
import random
import torch
from torch import Tensor
from torch.nn.functional import cosine_similarity


def client_selection(idxs_users, k, dev_user_loss, dev_user_time, args, m):
    # local_weights.to("cpu")
    # global_weights.to("cpu")
    alpha = 1.0  # 时间参数在选择概率中的比重
    et_sum = 0.0
    eq_sum = 0.0
    candidate = []
    t, q, p = {}, {}, []  # 时间偏差系数， 损失偏差系数， 概率
    count = 0
    # cosine_sum = 0
    # cosine = [torch.tensor(1.0) for i in range(args.user_num)]  # 全局只运行一次

    for idxs in range(args.user_num):
        if k[idxs] == 0:
            count += 1  # count ！= 0表示第一阶段
            candidate.append(idxs)  # 未参与过训练的客户端

    if count != 0:
        # 第一阶段：从未参与训练中的客户端选择
        print("第一阶段")
        if len(candidate) >= m:
            idxs_result = np.random.choice(candidate, m, replace=False)
        else:
            resurt = np.random.choice(list(set(range(args.user_num)) - set(candidate)), m - len(candidate), replace=False)
            idxs_result = np.append(candidate, resurt)
    else:
        print("第二阶段")
        # 第二阶段：根据概率选择
        candidate = range(args.user_num)
        dev_max_loss = max(dev_user_loss.values())
        dev_mim_loss = min(dev_user_loss.values())
        dev_max_time = max(dev_user_time.values())
        dev_mim_time = min(dev_user_time.values())
        for i in candidate:
            t[i] = (dev_user_time[i] - dev_mim_time) / (dev_max_time - dev_mim_time)  # 归一化时间和损失(min-max归一化)
            q[i] = (dev_user_loss[i] - dev_mim_loss) / (dev_max_loss - dev_mim_loss)
            et_sum += np.exp(-t[i])
            eq_sum += np.exp(-q[i])
        for i in candidate:
            if i != candidate[len(candidate) - 1]:
                pi = (np.exp(-t[i]) / et_sum) * alpha + (np.exp(-q[i]) / eq_sum) * (1.0 - alpha)  # 计算概率
                p.append(pi)
            else:
                p.append(1 - sum(p))
        idxs_result = np.random.choice(candidate, m, replace=False, p=p)
    return idxs_result


    # idxs_probability = []
    # for idx in range(args.user_num):  # 遍历每个客户端
    #     local_weights_v = []
    #     global_weights_v = []
    #
    #     for k, v in local_weights[idx].items():
    #         local_weights_v.append(v)  # 错了：local_weights_v应该是一个列表 [tensor([[[]),tensor([[[]),tensor([[[])...]
    #         # 列表中的每个元素是tensor()，经过这个循环之后local_weights_v是列表中最后一个tensor()
    #         # 解决方案： loacal_weights_v = []   local_weights_v.append(v)
    #         #           local_weights[idx]中的第i个v值添加到local_weights_v的第i个元素中
    #         #           计算余弦相似度也要对列表每个tensor数据计算，最后平均一下？
    #     for k, v in global_weights.items():
    #         global_weights_v.append(v)
    #
    #     if constraint[idx] == constraint_threshold:
    #         cosine[idx] = np.float32(1.0)
    #     elif constraint[idx] > 0:  # 如果约束因子为0，则移除客户端(令余弦相似度为0)
    #         cosine[idx] = np.float32(0.0)
    #         for i in range(len(local_weights_v)):
    #             # local_weights_v[i].to('cpu')
    #             # global_weights_v[i].to('cpu')
    #
    #             cos_layer = cosine_similarity(local_weights_v[i], global_weights_v[i], dim=-1).cpu()
    #             cos_layer = list(np.array(cos_layer).flatten())
    #
    #             cosine[idx] += (sum(cos_layer) / len(cos_layer))  # 第i层所有权重相似度的平均
    #
    #         cosine[idx] = cosine[idx] / len(local_weights_v)  # 平均所有层的相似度
    #         # 计算本地权重与全局权重之间的余弦相似度
    #     else:
    #         cosine[idx] = np.float32(0.0)
    #     cosine_sum += cosine[idx]  # 计算概率分母：余弦相似度总和
    #
    # for idx in range(args.user_num - 1):
    #     idxs_probability.append(cosine[idx] / cosine_sum)  # 计算客户端被选择的概率
    #
    # idxs_probability.append(1 - sum(idxs_probability))
    #
    # idxs_result = np.random.choice(range(args.user_num), m, replace=False, p=idxs_probability)  # 根据概率选择m个客户端

def client_selection_HSA_FL(idxs_users, k, com_power, args, m):
    # local_weights.to("cpu")
    # global_weights.to("cpu")
    # alpha = 0.5  # 时间参数在选择概率中的比重

    com_power_sum = 0.0
    count = 0
    candidate = []
    p = []

    for idxs in range(args.user_num):
        if k[idxs] == 0:
            count += 1  # count ！= 0表示第一阶段
            candidate.append(idxs)  # 未参与过训练的客户端

    if count != 0:
        print("第一阶段")
        # 第一阶段：从未参与训练中的客户端选择
        if len(candidate) >= m:
            idxs_result = np.random.choice(candidate, m, replace=False)
        else:
            resurt = np.random.choice(list(set(range(args.user_num)) - set(idxs_users) - set(candidate)), m - len(candidate), replace=False)
            idxs_result = np.append(candidate, resurt)
    else:
        print("第二阶段")
        for idxs in range(args.user_num):
            com_power_sum += com_power[idxs]

        for idxs in range(args.user_num - 1):
            p.append(com_power[idxs]/com_power_sum)
        p.append(1.0-sum(p))
        idxs_result = np.random.choice(range(args.user_num), m, replace=False, p=p)
    return idxs_result

def greedy_client_selection(user_loss, args):
    """

    :param user_loss: 用户的损失{idxs: loss}
    :param args:
    :return:
    """
    # user_loss_cluster_list = {i: {} for i in range(num_cluster)}    # key=集群号0-1-2-3   value={idxs: loss}
    #
    # m = max(int(args.frac * args.user_num), 1)  # 每轮训练客户端个数
    # for k, v in user_loss.items():
    #     user_loss_cluster_list[v[0]][k] = v[1]
    #
    # idxs_result = []
    # cluster_num = []    # 计算每个集群内客户端的个数
    # for i in range(len(user_loss_cluster_list)):
    #     cluster_num.append(len(user_loss_cluster_list[i]))

    ##########################################################
    greedy_para = 0.5
    m = max(int(args.frac * args.user_num), 1)  # 每轮训练客户端个数
    result = []
    s = sorted(user_loss.items(), key=lambda x: x[1], reverse=False)     # s是一个list[(),(),()]

    for i in range(int(m * greedy_para)):
        result.append(s[i][0])
    idxs_result2 = np.random.choice(list(set(range(args.user_num))-set(result)), int(m*(1-greedy_para)), replace=False)
    idxs_result = np.append(result, idxs_result2)
    print(idxs_result)
    return idxs_result





