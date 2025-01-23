import random


def set_local_epoch(idxs_users, users_time, users_loss, local_epoch, dev_user_time, dev_user_loss, users_level):
    avg_time = 0.0
    avg_loss = 0.0
    avg_epoch = 0

    for idxs in idxs_users:
        avg_time += users_time[idxs]
        avg_loss += users_loss[idxs]
        avg_epoch += local_epoch[idxs]
    avg_time = avg_time / len(idxs_users)
    avg_loss = avg_loss / len(idxs_users)
    avg_epoch = avg_epoch / len(idxs_users)
    for idxs in idxs_users:
        # 求每个客户端与平均的差值
        dev_user_time[idxs] = users_time[idxs] - avg_time
        dev_user_loss[idxs] = users_loss[idxs] - avg_loss
        if users_time[idxs] >= avg_time and users_loss[idxs] <= avg_loss:
            users_level[idxs] = 1  # 第一类，时间长精度高  降低迭代次数
            epoch_adjust = max(int((dev_user_time[idxs] / users_time[idxs]) * local_epoch[idxs]), 1)
            # if local_epoch[idxs] > epoch_adjust:
            #     local_epoch[idxs] -= epoch_adjust
            if local_epoch[idxs] > epoch_adjust + 5:
                local_epoch[idxs] -= epoch_adjust
            else:
                local_epoch[idxs] = 5

        elif users_time[idxs] <= avg_time and users_loss[idxs] >= avg_loss:
            users_level[idxs] = 2  # 第二类，时间短精度低  增加迭代次数
            epoch_adjust = max(int((dev_user_loss[idxs] / avg_loss) * avg_epoch), 1)

            # local_epoch[idxs] += epoch_adjust         ###################
            if local_epoch[idxs] < 13:
                local_epoch[idxs] += 1
            else:
                local_epoch[idxs] = 13

        elif users_time[idxs] <= avg_time and users_loss[idxs] <= avg_loss:
            users_level[idxs] = 3  # 第三类，时间短精度高  保持不变迭代次数

        else:
            users_level[idxs] = 4  # 第四类，时间长精度低(劣质user)  保持不变迭代次数

    return users_level, local_epoch, dev_user_time, dev_user_loss


if __name__ == '__main__':
    idxs_users = range(10)
    users_time, users_acc, local_epoch, dev_user_time, dev_user_acc, users_level = [], [], [], {}, {}, []
    for i in idxs_users:
        users_time.append(random.randint(1, 10) * 0.123456789)
        users_acc.append(random.randint(1, 7) * 0.123456789)
        local_epoch.append(10)
        dev_user_time[i] = 0.0
        dev_user_acc[i] = 0.0
        users_level.append(0)
    users_level, local_epoch, dev_user_time, dev_user_acc = set_local_epoch(idxs_users, users_time, users_acc,
                                                                            local_epoch, dev_user_time,
                                                                            dev_user_acc, users_level)
    print(users_time)
    print(users_acc)
    print(users_level)
    print(local_epoch)
    print(dev_user_time)
    print(dev_user_acc)
