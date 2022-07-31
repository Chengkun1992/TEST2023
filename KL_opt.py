# -- coding: utf-8 --
import torch
import torch.nn as nn
import numpy as np
import time
import math
import cProfile
from sklearn.metrics import precision_recall_fscore_support, precision_recall_curve, f1_score, roc_curve, auc
from multiprocessing import Pool as tp
import pandas as pd


def generate_test_data(num, dim):
    data = np.random.randn(num, dim)
    # print(data)
    return data


def tensorlize_array(list_):
    b = torch.from_numpy(list_).float()
    return b


def tensorlize_list(list_):
    # list_ = list_.astype(float)
    a = np.array(list_)
    b = torch.from_numpy(a).float()
    return b


class JSD(nn.Module):
    def __init__(self):
        super(JSD, self).__init__()
        self.kl = nn.KLDivLoss(reduction='sum', log_target=True)

    def forward(self, p: torch.tensor, q: torch.tensor):
        # print(p.size())
        p, q = p.view(-1, p.size(-1)), q.view(-1, q.size(-1))
        # print(p.size())
        m = (0.5 * (p + q)).log()
        return 0.5 * (self.kl(m, p.log()) + self.kl(m, q.log()))


def no_opt_JSD(gt, pre):
    crition_JS = JSD()
    num = gt.shape[0]
    value = 0
    for i in range(num):
        ten1 = gt[i]
        ten2 = pre[i]
        st = time.time()
        loss = crition_JS(ten1, ten2)
        # loss = JSD_(ten1, ten2)
        tc = time.time() - st
        # print('time cost: ', tc, loss)
        value += tc
    print(value, value / num)


def self_JSD(gt, pre):
    num = gt.shape[0]
    value = 0
    losses = []
    for i in range(num):
        ten1 = gt[i]
        ten2 = pre[i]
        st = time.time()
        loss = JSD_(ten1, ten2)
        # norm_ = np.sum(np.abs(ten1 - ten2))
        # up_b = 0.5 * norm_
        #
        # if loss > up_b:
        #     print('yes!')

        tc = time.time() - st
        # print('time cost: ', tc, loss)
        losses.append(loss / 400)
        value += tc
    # print('JSD time cost: ', value, value / num)
    return np.array(losses), value


def MSE_(v1, v2):
    sum_ = 0
    for y1, y2 in zip(v1, v2):
        sum_ += (y2 - y1) * (y2 - y1)
    return sum_


def self_MSE(gt, pre):
    num = gt.shape[0]
    value = 0
    losses = []
    for i in range(num):
        ten1 = gt[i]
        ten2 = pre[i]
        st = time.time()
        loss = MSE_(ten1, ten2)
        tc = time.time() - st
        # print('time cost: ', tc, loss)
        losses.append(loss)
        value += tc
    # print('MSE time cost: ', value, value / num)
    return np.array(losses), value


def L1_norm_self(v1, v2, threshold):
    # v0 = tensorlize_array(v1)
    # arr_1, ind = torch.sort(v0, -1, True)
    # for i in ind:
    #     sum_ += abs(v1[i] - v2[i]) / 400
    #     if sum_ > threshold:
    #         # print('early stop')
    #         sum_ = 0
    #         break
    sum_ = np.mean(np.abs(v1 - v2))
    if sum_ > threshold:
        sum_ = 0  # low bound > ths
    return sum_


def Pinsker(P, Q):
    norm_ = np.sum(np.abs(P - Q))
    return 0.5 * norm_ * norm_


def L1_norm_2(v1, v2, threshold):
    v3 = (v1 + v2) / 2
    low_val = (Pinsker(v1, v3) + Pinsker(v2, v3)) * 0.5
    if low_val / 400 > threshold:
        return 1  # larger than threhold → novel
    else:
        return 0


def test_up_L_norm(v1, v2, threshold):
    norm_ = np.sum(np.abs(v1 - v2))
    up_b = 0.5 * norm_
    flag = False
    if up_b / 400 < threshold:
        flag = True
    return up_b, flag


def find_bounds(v1, v2):
    norm_ = np.sum(np.abs(v1 - v2))
    up_b = 0.5 * norm_
    v3 = (v1 + v2) / 2
    low_b = (Pinsker(v1, v3) + Pinsker(v2, v3)) * 0.5
    return low_b / 400, up_b / 400


def self_L1(gt, pre, threshold):
    num = gt.shape[0]
    value = 0
    for i in range(num):
        ten1 = gt[i]
        ten2 = pre[i]
        st = time.time()
        loss = L1_norm_self(ten1, ten2, threshold)
        tc = time.time() - st
        # print('time cost: ', tc, loss)
        value += tc
    print(value, value / num)


def L1_norm(gt, pre):
    crition_L = nn.L1Loss(reduction='sum')
    num = gt.shape[0]
    value = 0
    for i in range(num):
        ten1 = gt[i]
        ten2 = pre[i]
        st = time.time()
        loss = crition_L(ten1, ten2)
        tc = time.time() - st
        print('time cost: ', tc, loss)
        value += tc
    print(value, value / num)


def KLD(P, Q):
    sum = 0
    s1 = 0
    s2 = 0
    for p, q in zip(P, Q):
        # print(p, q)
        item = p * np.log(p / q)
        # print(item)
        # s1 += p
        # s2 += q
        sum += item
    # print(sum, s1, s2)
    return sum


def JSD_(P, Q):
    target = (P + Q) / 2
    sum_ = (KLD(P, target) + KLD(Q, target)) / 2
    return sum_


def JSD_hash(P, Q, T, count):
    sum_ = 0
    for p, q, t, c in zip(P, Q, T, count):
        item_1 = p * np.log(p / t) * c
        item_2 = q * np.log(q / t) * c
        sum_ += (item_1 + item_2) / 2
    return sum_


def hash_multi_1(val):
    if val >= 0.1:
        coded = [1]
    else:
        coded = [0]
    return coded


def hash_multi_2(val):
    if val >= 0.1:
        coded = [0, 0]
    else:
        if val >= 1e-2:
            coded = [0, 1]
        else:
            if val >= 1e-2:
                coded = [1, 0]
            else:
                coded = [1, 1]
    return coded


def hash_match_1(v1, v2, threshold):
    x = np.where(v1 > threshold, 1, 0)
    y = np.where(v2 > threshold, 1, 0)
    # print(x, y)
    res = np.bitwise_xor(x, y)
    # print(res[:10])
    # print(res, res[res > 0].shape[0])
    if res.max() == 1:
        # print('1',[res== 'True'])
        return False, x, y
    return True, x, y


def compass(x, v):
    num = x.shape[-1]
    tmp_seg = []
    new_ = []
    max_ = []
    min_ = []
    max_v = -1
    min_v = 100000
    input = 0
    count_ = []
    for i in range(num):
        if x[i] == 1:
            if input:
                new_.append(0)
                max_.append(max_v)
                min_.append(min_v)
                count_.append(len(tmp_seg))
                tmp_seg = []
                max_v = -1
                min_v = 100000
                input = 0
            new_.append(1)
            # torch.cat((max_, v[i]))
            # torch.cat((min_, v[i]))
            max_.append(v[i])
            min_.append(v[i])
            count_.append(1)
        else:
            input = 1
            tmp_seg.append(v[i])
            # torch.cat((max_, v[i]))
            # torch.cat((min_, v[i]))
            max_v = max(max_v, v[i])
            min_v = min(min_v, v[i])
    if input:
        new_.append(0)
        # torch.cat((max_, v[i]))
        # torch.cat((min_, v[i]))
        max_.append(max_v)
        min_.append(min_v)
        count_.append(len(tmp_seg))
    # print(new_, count_)
    # print(torch.stack(max_))
    # print(torch.stack(min_))
    return np.array(max_), np.array(min_), np.array(count_)


def compress_2(hash_, vec):
    com = vec[np.where(hash_ == 0)]
    ori = vec[np.where(hash_ == 1)]
    # print(ori, type(ori), ori.shape)
    return com, ori


def compress_3(hash_, vec):
    ori = vec[np.where(hash_ == 1)]
    return ori


def UpperBound_2(v1, v2, threshold):
    x1 = np.where((v1 >= 1e-1), np.ones_like(v1, dtype=int), np.zeros_like(v1, dtype=int))
    x2 = np.where((1e-1 > v1) & (v1 >= 1e-2), np.ones_like(v1, dtype=int), np.zeros_like(v1, dtype=int))
    x3 = np.where((1e-4 > v1) & (v1 >= 1e-3), np.ones_like(v1, dtype=int), np.zeros_like(v1, dtype=int))
    x4 = np.where((1e-3 > v1) & (v1 >= 1e-4), np.ones_like(v1, dtype=int), np.zeros_like(v1, dtype=int))
    x5 = np.where((1e-4 > v1) & (v1 >= 1e-6), np.ones_like(v1, dtype=int), np.zeros_like(v1, dtype=int))
    x6 = np.where((1e-6 > v1) & (v1 >= 1e-8), np.ones_like(v1, dtype=int), np.zeros_like(v1, dtype=int))
    x7 = np.where((1e-8 > v1) & (v1 >= 1e-10), np.ones_like(v1, dtype=int), np.zeros_like(v1, dtype=int))

    y1 = np.where((v2 >= 1e-1), np.ones_like(v2, dtype=int), np.zeros_like(v2, dtype=int))
    y2 = np.where((1e-1 > v2) & (v2 >= 1e-2), np.ones_like(v2, dtype=int), np.zeros_like(v2, dtype=int))
    y3 = np.where((1e-2 > v2) & (v2 >= 1e-3), np.ones_like(v2, dtype=int), np.zeros_like(v2, dtype=int))
    y4 = np.where((1e-3 > v2) & (v2 >= 1e-4), np.ones_like(v2, dtype=int), np.zeros_like(v2, dtype=int))
    y5 = np.where((1e-4 > v2) & (v2 >= 1e-6), np.ones_like(v2, dtype=int), np.zeros_like(v2, dtype=int))
    y6 = np.where((1e-6 > v2) & (v2 >= 1e-8), np.ones_like(v2, dtype=int), np.zeros_like(v2, dtype=int))
    y7 = np.where((1e-8 > v2) & (v2 >= 1e-10), np.ones_like(v2, dtype=int), np.zeros_like(v2, dtype=int))
    f1 = np.bitwise_xor(x1, y1)
    f2 = np.bitwise_xor(x2, y2)
    f3 = np.bitwise_xor(x3, y3)
    # flag = f1.max() + f2.max() + f3.max()
    flag = 0
    if flag > 0:
        return False
    else:
        loss = JSD_(v1, v2) / 400
        if loss > threshold:
            fg = 1
        else:
            fg = 0
        ori_1 = compress_3(x1, v1)
        ori_2 = compress_3(x2, v1)
        ori_3 = compress_3(x3, v1)
        ori_4 = compress_3(x4, v1)
        ori_5 = compress_3(x5, v1)
        ori_6 = compress_3(x6, v1)
        ori_7 = compress_3(x7, v1)

        ab = [ori_1.shape[0], ori_2.shape[0], ori_3.shape[0], ori_4.shape[0], ori_5.shape[0], ori_6.shape[0],
              ori_7.shape[0]]
        print('1 length', ab, sum(ab), fg)
        ori_1 = compress_3(y1, v2)
        ori_2 = compress_3(y2, v2)
        ori_3 = compress_3(y3, v2)
        ori_4 = compress_3(y4, v2)
        ori_5 = compress_3(y5, v2)
        ori_6 = compress_3(y6, v2)
        ori_7 = compress_3(y7, v2)
        ab = [ori_1.shape[0], ori_2.shape[0], ori_3.shape[0], ori_4.shape[0], ori_5.shape[0], ori_6.shape[0],
              ori_7.shape[0]]
        print('2 length', ab, sum(ab), fg)

        return True


def UpperBound_2_2(v1, v2, threshold):
    # x1 = np.where((v1 >= 1/2), np.ones_like(v1, dtype=int), np.zeros_like(v1, dtype=int))
    # x2 = np.where((1e-1 > v1) & (v1 >= 1e-2), np.ones_like(v1, dtype=int), np.zeros_like(v1, dtype=int))
    # x3 = np.where((1e-4 > v1) & (v1 >= 1e-3), np.ones_like(v1, dtype=int), np.zeros_like(v1, dtype=int))
    # x4 = np.where((1e-3 > v1) & (v1 >= 1e-4), np.ones_like(v1, dtype=int), np.zeros_like(v1, dtype=int))
    # x5 = np.where((1e-4 > v1) & (v1 >= 1e-6), np.ones_like(v1, dtype=int), np.zeros_like(v1, dtype=int))
    # x6 = np.where((1e-6 > v1) & (v1 >= 1e-8), np.ones_like(v1, dtype=int), np.zeros_like(v1, dtype=int))
    # x7 = np.where((1e-8 > v1) & (v1 >= 1e-10), np.ones_like(v1, dtype=int), np.zeros_like(v1, dtype=int))
    #
    # y1 = np.where((v2 >= 1e-1), np.ones_like(v2, dtype=int), np.zeros_like(v2, dtype=int))
    # y2 = np.where((1e-1 > v2) & (v2 >= 1e-2), np.ones_like(v2, dtype=int), np.zeros_like(v2, dtype=int))
    # y3 = np.where((1e-2 > v2) & (v2 >= 1e-3), np.ones_like(v2, dtype=int), np.zeros_like(v2, dtype=int))
    # y4 = np.where((1e-3 > v2) & (v2 >= 1e-4), np.ones_like(v2, dtype=int), np.zeros_like(v2, dtype=int))
    # y5 = np.where((1e-4 > v2) & (v2 >= 1e-6), np.ones_like(v2, dtype=int), np.zeros_like(v2, dtype=int))
    # y6 = np.where((1e-6 > v2) & (v2 >= 1e-8), np.ones_like(v2, dtype=int), np.zeros_like(v2, dtype=int))
    # y7 = np.where((1e-8 > v2) & (v2 >= 1e-10), np.ones_like(v2, dtype=int), np.zeros_like(v2, dtype=int))
    xs = []
    ys = []
    val = 1
    flags = []
    seg_number = 18  # 2**(-seg_number-1)
    step = 2
    for i in range(seg_number):
        if i == 0:
            val = 1 / step
            x = np.where((v1 >= val), np.ones_like(v1, dtype=int), np.zeros_like(v1, dtype=int))
            xs.append(x)
            y = np.where((v2 >= val), np.ones_like(v2, dtype=int), np.zeros_like(v2, dtype=int))
            ys.append(y)
            f = np.bitwise_xor(x, y)
            flags.append(f.max())
            continue
        if i == seg_number - 1:
            x = np.where((v1 < val), np.ones_like(v1, dtype=int), np.zeros_like(v1, dtype=int))
            xs.append(x)
            y = np.where((v2 < val), np.ones_like(v2, dtype=int), np.zeros_like(v2, dtype=int))
            ys.append(y)
            f = np.bitwise_xor(x, y)
            flags.append(f.max())
            # print('last: ', val)
            break
        x = np.where((val > v1) & (v1 >= val / step), np.ones_like(v1, dtype=int), np.zeros_like(v1, dtype=int))
        y = np.where((val > v2) & (v2 >= val / step), np.ones_like(v2, dtype=int), np.zeros_like(v2, dtype=int))
        f = np.bitwise_xor(x, y)
        flags.append(f.max())
        xs.append(x)
        ys.append(y)
        val = val / step
    # f1 = np.bitwise_xor(x1, y1)
    # f2 = np.bitwise_xor(x2, y2)
    # f3 = np.bitwise_xor(x3, y3)
    # flag = f1.max() + f2.max()  # + f3.max()

    if 0:
        return False
    else:
        # loss = JSD_(v1, v2) / 400
        # if loss > threshold:
        #     fg = 1
        # else:
        #     fg = 0
        ori_x = []
        abx = []
        ori_y = []
        aby = []
        for i in range(seg_number):
            ori = compress_3(xs[i], v1)
            ori_x.append(ori)
            abx.append(ori.shape[0])
            ori = compress_3(ys[i], v2)
            ori_y.append(ori)
            aby.append(ori.shape[0])

        print('gt: ', abx, sum(abx), flags)
        print('pre: ', aby, sum(aby))

        return flags


def UpperBound(v1, v2, threshold, thresholds_hash):
    st = time.time()
    flag, x, y = hash_match_1(v1, v2, thresholds_hash)
    hash_t = time.time() - st
    com_time = 0
    if flag:
        # print('hash match!')
        # ---------------------
        # max_1, min_1, count_1 = compass(x, v1)
        # max_2, min_2, count_2 = compass(y, v2)
        # min_3 = (min_1 + min_2) / 2
        # loss = JSD_hash(max_1, max_2, min_3, count_1) / 400
        # ---------------------
        com_1, ori_1 = compress_2(x, v1)
        com_2, ori_2 = compress_2(y, v2)
        max_1 = com_1.max()
        max_2 = com_2.max()
        min_3 = ((com_1 + com_2) / 2).min()
        length = com_2.shape[0]
        # print('0 length: ', length)
        st = time.time()
        loss = JSD_(ori_2, ori_1) + (max_1 * np.log(max_1 / min_3) + max_2 * np.log(max_2 / min_3)) * length / 2
        com_time = time.time() - st
        print('true diff: ', loss - JSD_(v1, v2))
        print('extra', max_1, max_2, min_3, com_1.mean(), com_2.mean())
        if loss / (400) < threshold:
            print('hit!')
            return 1, hash_t, com_time
        else:
            return 0, hash_t, com_time

    else:
        return 0, hash_t, com_time


def boundaries(gt, pre, thresholds):
    num = gt.shape[0]
    thresholds_hash = []
    for i in range(num):
        thresholds_hash.append(gt[i].max() / 5)

    pre_index = []
    count_low = 0
    count_up = 0

    low_time = 0
    up_time = 0
    waste_time = 0
    st = time.time()
    for i in range(num):
        t1 = time.time()
        flag = L1_norm_self(gt[i], pre[i], thresholds[i])
        low_time += time.time() - t1
        if flag == 0:
            pre_index.append(0)
            count_low += 1
        else:
            t1 = time.time()
            flag_2 = UpperBound(gt[i], pre[i], thresholds[i], thresholds_hash[i])
            up_time += time.time() - t1
            if flag_2 == 1:
                pre_index.append(1)
                count_up += 1
            else:
                t1 = time.time()
                loss = JSD_(gt[i], pre[i])
                waste_time += time.time() - t1
                if loss > thresholds[i]:
                    pre_index.append(1)
                else:
                    pre_index.append(0)
    t = time.time() - st
    print(count_low, count_up)
    print(low_time, up_time, waste_time)
    return sum([low_time, up_time, waste_time])


def boundaries_L1(gt, pre, thresholds):
    num = gt.shape[0]

    pre_index = []
    count_low = 0
    count_up = 0

    low_time = 0
    up_time = 0
    waste_time = 0
    for i in range(num):
        t1 = time.time()
        flag = L1_norm_2(gt[i], pre[i], thresholds[i])
        # flag = 1
        low_time += time.time() - t1
        if flag == 1:  # low bound larger than threshold
            pre_index.append(1)
            count_low += 1
        else:
            t1 = time.time()
            loss = JSD_(gt[i], pre[i])
            waste_time += time.time() - t1
            # if loss > thresholds[i]:
            #     pre_index.append(1)
            # else:
            #     pre_index.append(0)

    print(count_low, count_up)
    print(low_time, up_time, waste_time)
    return sum([low_time, up_time, waste_time])


def boundaries_L1_(gt, pre, thresholds):
    num = gt.shape[0]
    thresholds_hash = []
    # for i in range(num):
    #     thresholds_hash.append(gt[i].max() / 5)

    pre_index = []
    count_low = 0
    count_up = 0

    low_time = 0
    up_time = 0
    waste_time = 0
    for i in range(num):
        t1 = time.time()
        flag = L1_norm_self(gt[i], pre[i], thresholds[i])
        low_time += time.time() - t1
        if flag == 0:
            pre_index.append(0)
            count_low += 1
        else:
            t1 = time.time()
            loss = JSD_(gt[i], pre[i])
            waste_time += time.time() - t1
            if loss > thresholds[i]:
                pre_index.append(1)
            else:
                pre_index.append(0)
    print(count_low, count_up)
    print(low_time, up_time, waste_time)
    return sum([low_time, up_time, waste_time])


def boundaries_upper(gt, pre, thresholds):
    num = gt.shape[0]
    thresholds_hash = []
    for i in range(num):
        thresholds_hash.append(gt[i].max() / 10)

    pre_index = []
    count_low = 0
    count_up = 0
    hash_time = 0
    fast_JS_time = 0
    low_time = 0
    up_time = 0
    waste_time = 0
    st = time.time()
    for i in range(num):
        t1 = time.time()
        flag = flag = L1_norm_self(gt[i], pre[i], thresholds[i])
        low_time += time.time() - t1
        if flag == 0:
            pre_index.append(1)
            count_low += 1
        else:
            t1 = time.time()
            flag_2, hash_t, com_t = UpperBound(gt[i], pre[i], thresholds[i], thresholds_hash[i])
            up_time += time.time() - t1
            hash_time += hash_t
            fast_JS_time += com_t
            if flag_2 == 1:
                pre_index.append(0)
                count_up += 1
            else:
                t1 = time.time()
                loss = JSD_(gt[i], pre[i])
                waste_time += time.time() - t1
                if loss > thresholds[i]:
                    pre_index.append(1)
                else:
                    pre_index.append(0)
    t = time.time() - st
    print(count_low, count_up)
    print('hash time: ', hash_time, 'hash_compute: ', fast_JS_time)
    print(low_time, up_time, waste_time)
    return sum([low_time, up_time, waste_time])


def boundaries_upper_2(gt, pre, thresholds):
    num = gt.shape[0]
    # thresholds_hash = []
    # for i in range(num):
    #     thresholds_hash.append(gt[i].max() / 10)
    # print('all', gt.max(), gt.min())
    # print('all', pre.max(), pre.min())
    pre_index = []
    count_low = 0
    count_up = 0
    hash_time = 0
    fast_JS_time = 0
    low_time = 0
    up_time = 0
    waste_time = 0
    st = time.time()
    for i in range(num):
        # t1 = time.time()
        # flag = 1
        # low_time += 0
        if 0:
            pre_index.append(1)
            count_low += 1
        else:
            t1 = time.time()
            flag2 = UpperBound_2(gt[i], pre[i], thresholds[i])
            up_time += time.time() - t1
            if flag2:
                count_up += 1
    print(count_low, count_up)
    # print('hash time: ', hash_time, 'hash_compute: ', fast_JS_time)
    # print('----default----upbound-----JSD----')
    # print(low_time, up_time, waste_time)
    return sum([low_time, up_time, waste_time])


def gene_dict(order):
    num = 2 ** order
    dic = {0: 0}
    for i in range(num):
        dic.update({i + 1: i})
    # print(dic)
    return dic


def boundaries_upper_order(gt, pre, thresholds):
    num = gt.shape[0]
    pre_index = []
    count_low = 0
    count_up = 0
    hash_time = 0
    fast_JS_time = 0
    low_time = 0
    up_time = 0
    up_time_0 = 0
    waste_time = 0
    dic = gene_dict(1)
    for i in range(num):
        # t1 = time.time()
        # flag = 1
        # low_time += 0
        if 0:
            pre_index.append(1)
            count_low += 1
        else:
            t1 = time.time()
            flag2 = UpperBound_order(gt[i], pre[i], thresholds[i], 1)
            # up_time_0 += time.time() - t1
            # t1 = time.time()
            # flag2, h_time, com_time = UpperBound_new(gt[i], pre[i], thresholds[i], dic)
            up_time += time.time() - t1
            # hash_time += h_time
            # fast_JS_time += com_time
            if flag2:
                count_up += 1
            else:
                t1 = time.time()
                loss = JSD_(gt[i], pre[i])
                waste_time += time.time() - t1

    print(count_low, count_up)
    print('hash time: ', hash_time, 'hash_compute: ', fast_JS_time)
    print('----default----upbound-----JSD----')
    print(low_time, up_time_0, up_time, waste_time)
    return sum([low_time, up_time, waste_time])


def boundaries_upper_new(gt, pre, thresholds):
    num = gt.shape[0]
    count_low = 0
    count_up = 0
    hash_time = 0
    fast_JS_time = 0
    low_time = 0
    up_time = 0
    waste_time = 0
    match_count = 0
    dic = gene_dict(order_)
    for i in range(num):
        # t1 = time.time()
        # flag = 1
        # low_time += 0
        # if i != 13 and i != 24:
        #     continue
        v1 = pre[i]
        v2 = gt[i]
        max_v = np.max([v1.max(), v2.max()])
        min_v = np.min([v1.min(), v2.min()])
        bw = (max_v - min_v) / (len(dic) - 1)
        seg_mm = []
        hashed_hist = []
        for j in range(len(dic) - 1):
            mx = (j + 1) * bw + min_v
            mi = j * bw + min_v
            seg_mm.append([1, -1])
            # seg_mm.append([mi, mx])
            hashed_hist.append(0)
        opt = [dic, seg_mm, hashed_hist, max_v, min_v, bw]
        if 0:
            pre_index.append(1)
            count_low += 1
        else:
            t1 = time.time()
            flag2, h_time, com_time, is_match = UpperBound_new(gt[i], pre[i], thresholds[i], opt)
            match_count += is_match
            up_time += time.time() - t1
            hash_time += h_time
            fast_JS_time += com_time
            # print('flag2', flag2)
            up_b, flag3 = test_up_L_norm(gt[i], pre[i], thresholds[i])
            if flag2 or flag3:
                count_up += 1
            # if flag2:
            # print('??', count_up)
            # print(i)
            # break
            # count_up += 1
            else:
                t1 = time.time()
                loss = JSD_(gt[i], pre[i])
                waste_time += time.time() - t1
        # break
    print('lower bound filter: ', count_low)
    print('upper bound filter: ', count_up)
    print('matched ', match_count)
    print('hash time: ', hash_time, 'hash_compute: ', fast_JS_time)
    print('----default----upbound-----JSD----')
    print(low_time, up_time, waste_time)
    return sum([low_time, up_time, waste_time])


def hash_func(vec, max_v, min_v, order):
    hist_num = 2 ** order
    bw = (max_v - min_v) / 8
    dict_ = 1
    for ele in vec:
        ele


def s_xor(v1, v2):
    # res = [(ord(x) ^ ord(y)) for x, y in zip(v1, v2)]
    flag = 1
    for x, y in zip(v1, v2):
        tmp = ord(x) ^ ord(y)
        print(tmp)
        if tmp == 1:
            flag = 0
            break
    return flag


def hash_match_new(v1, v2, opt):
    dic, seg_mm, hashed_hist, max_v, min_v, bw = opt
    # print(max_v, min_v)
    # print(bw)
    # print('------------')
    # max_v = max(v1.max(), v2.max())
    # min_v = min(v1.min(), v2.min())
    # bw = (max_v - min_v) / (len(dic) - 1)
    # print(max_v, min_v)
    # print(bw)
    # seg_mm = []
    # hashed_hist = []
    # for i in range(len(dic) - 1):
    #     mx = (i + 1) * bw + min_v
    #     mi = i * bw + min_v
    #     seg_mm.append([mi, mx])
    #     hashed_hist.append(0)
    is_match = 1

    # for x, y in zip(v1, v2):
    #     hash_x = math.ceil((x - min_v) / bw)
    #     hash_y = math.ceil((y - min_v) / bw)
    #     hash_x = dic.get(hash_x, -1)
    #     hash_y = dic.get(hash_y, -1)
    #     if hash_x != hash_y:
    #         is_match = 0
    #         # print('early stop!')
    #         break
    #     hashed_hist[hash_x] += 1
    #     seg_mm[hash_x][0] = np.min([np.min([seg_mm[hash_x][0], x]), y])
    #     seg_mm[hash_x][1] = np.max([np.max([seg_mm[hash_x][1], x]), y])
    # print(is_match)
    # print('old')
    # print(hashed_hist)
    # print(seg_mm)
    # loss = 0
    # for i in range(len(seg_mm)):
    #     q_min = seg_mm[i][0]
    #     p_max = seg_mm[i][1]
    #     loss += (p_max * np.log(p_max / q_min)) * hashed_hist[i]
    # print('loss', loss)
    # print('old')
    #
    # is_match = 1
    v1_ = np.ceil((v1 - min_v) / bw)
    v2_ = np.ceil((v2 - min_v) / bw)
    _v1_ = np.where(v1_ == 0, 1, v1_)
    _v2_ = np.where(v2_ == 0, 1, v2_)
    res = np.bitwise_xor(_v1_.astype(int), _v2_.astype(int))
    # print(res)
    if res.max() > 0:
        is_match = 0
    else:
        # print('segmm ', len(seg_mm))
        for i in range(len(seg_mm)):
            v_ind = np.where(_v1_ == (i + 1))
            v_ind_2 = np.where(_v2_ == (i + 1))
            hashed_hist[i] = v_ind[0].shape[0]
            if v_ind[0].shape[0] == 0:
                seg_mm[i] = [1, 1]
                continue

            # v2_ind = np.where(_v2_ == (i+1))
            t1 = v1[v_ind]
            t2 = v2[v_ind_2]
            # if i == 1:
            # print(v_ind, v_ind_2)
            # print(t1)
            # print(np.ceil((t1 - min_v) / bw))
            # print(t2)
            # print(np.ceil((t2 - min_v) / bw))
            # print(t1.max(), t2.max())
            # tnp = np.max([t1.max(), t2.max()])
            # print(tnp, type(tnp))
            # #
            # tnp = np.min([t1.min(), t2.min()])
            # print(tnp, type(tnp))
            seg_mm[i][1] = np.max([t1.max(), t2.max()])
            seg_mm[i][0] = np.min([t1.min(), t2.min()])
    # print('new')
    # print(is_match)
    # print(hashed_hist)
    # print(seg_mm)
    # loss = 0
    # for i in range(len(seg_mm)):
    #     q_min = seg_mm[i][0]
    #     p_max = seg_mm[i][1]
    #     loss += (p_max * np.log(p_max / q_min)) * hashed_hist[i]
    # print('loss', loss)
    # print('new')
    return is_match, hashed_hist, seg_mm


def UpperBound_new(v1, v2, threshold, opt):
    # print('New bound!')
    com_time = 0
    dic, seg_mm, hashed_hist, max_v, min_v, bw = opt
    # v1_ = v1 - min_v
    # v2_ = v2 - min_v
    t0 = time.time()
    flag, hist, seg_mm = hash_match_new(v1, v2, opt)
    hash_time = time.time() - t0
    x1 = 1
    y1 = 1
    if flag == 0:  # hashed vectors do not match
        # print('??! dismatch')
        return 0, hash_time, com_time, flag
    else:
        # print('!!! match')
        loss = 0
        st = time.time()

        up_b, flag3 = test_up_L_norm(v1, v2, threshold)

        for i in range(len(seg_mm)):
            q_min = seg_mm[i][0]
            p_max = seg_mm[i][1]
            loss += (p_max * np.log(p_max / q_min)) * hist[i]
            # if loss / 400 > threshold:
            #     return False, hash_time, com_time
        com_time = time.time() - st
        # print('com', com_time)
        # print('true diff: ', loss - JSD_(v1, v2))
        # print('extra', max_1, max_2, min_3, com_1.mean(), com_2.mean())
        # if loss / 400 < threshold:
        if flag3:
            # print('hit!', end='')
            # loss_ = JSD_(v1, v2) / 400
            # if loss_>=threshold:
            #     print('wtf!')
            # print(loss/400, loss_)
            # print('hit!---------')
            # print(hist)
            # print(seg_mm)
            return 1, hash_time, com_time, flag
        else:
            # loss = JSD_(v1, v2) / 400
            # if loss > threshold:
            #     fg = 1
            # else:
            #     fg = 0
            return 0, hash_time, com_time, flag


def UpperBound_order(v1, v2, threshold, order):
    f1 = np.array([0])
    f2 = np.array([0])
    f3 = np.array([0])
    if order == 1:
        x1 = np.where((v1 >= 1e-1), np.ones_like(v1, dtype=int), np.zeros_like(v1, dtype=int))
        y1 = np.where((v2 >= 1e-1), np.ones_like(v2, dtype=int), np.zeros_like(v2, dtype=int))
        f1 = np.bitwise_xor(x1, y1)
        flag = f1.max()
    if order == 2:
        x1 = np.where((v1 >= 1e-2), np.ones_like(v1, dtype=int), np.zeros_like(v1, dtype=int))
        y1 = np.where((v2 >= 1e-2), np.ones_like(v2, dtype=int), np.zeros_like(v2, dtype=int))
        f2 = np.bitwise_xor(x1, y1)
        flag = f2.max()
    if order == 3:
        x1 = np.where((v1 >= 1e-3), np.ones_like(v1, dtype=int), np.zeros_like(v1, dtype=int))
        y1 = np.where((v2 >= 1e-3), np.ones_like(v2, dtype=int), np.zeros_like(v2, dtype=int))
        f3 = np.bitwise_xor(x1, y1)
        flag = f3.max()
    if flag > 0:  # hashed vectors do not match
        return False
    else:
        com_1, ori_1 = compress_2(x1, v1)
        com_2, ori_2 = compress_2(y1, v2)
        max_1 = com_1.max()
        max_2 = com_2.max()
        min_3 = ((com_1 + com_2) / 2).min()
        length = com_2.shape[0]
        st = time.time()
        loss = JSD_(ori_2, ori_1) + (max_1 * np.log(max_1 / min_3) + max_2 * np.log(max_2 / min_3)) * length / 2
        com_time = time.time() - st
        # print('true diff: ', loss - JSD_(v1, v2))
        # print('extra', max_1, max_2, min_3, com_1.mean(), com_2.mean())
        if loss / 400 < threshold:
            print('hit!')
            return True
        else:
            # loss = JSD_(v1, v2) / 400
            # if loss > threshold:
            #     fg = 1
            # else:
            #     fg = 0
            return False


def Bounds_new(gt, pre, thresholds):
    num = gt.shape[0]
    pre_index = []
    count_low = 0
    count_up = 0
    low_time = 0
    up_time = 0
    waste_time = 0
    match_count = 0
    dic = gene_dict(order_)
    for i in range(num):
        t1 = time.time()
        flag = L1_norm_2(gt[i], pre[i], thresholds[i])
        # flag = 1
        low_time += time.time() - t1
        if flag == 1:  # low bound larger than threshold
            pre_index.append(1)
            count_low += 1
        else:
            v1 = pre[i]
            v2 = gt[i]
            max_v = np.max([v1.max(), v2.max()])
            min_v = np.min([v1.min(), v2.min()])
            bw = (max_v - min_v) / (len(dic) - 1)
            seg_mm = []
            hashed_hist = []
            for j in range(len(dic) - 1):
                mx = (j + 1) * bw + min_v
                mi = j * bw + min_v
                seg_mm.append([1, -1])
                # seg_mm.append([mi, mx])
                hashed_hist.append(0)
            opt = [dic, seg_mm, hashed_hist, max_v, min_v, bw]
            t1 = time.time()
            flag2, hash_time, com_time, is_match = UpperBound_new(gt[i], pre[i], thresholds[i], opt)
            up_time += time.time() - t1
            match_count += is_match
            # up_b, flag3 = test_up_L_norm(gt[i], pre[i], thresholds[i])
            if flag2:  # or flag3:
                count_up += 1
            else:
                t1 = time.time()
                loss = JSD_(gt[i], pre[i])

                waste_time += time.time() - t1
            # if loss > thresholds[i]:
            #     pre_index.append(1)
            # else:
            #     pre_index.append(0)
    print('lowbound', count_low, 'upbound', count_up)
    print('match case: ', match_count)
    print(low_time, up_time, waste_time)
    return sum([low_time, up_time, waste_time])


def Bounds_L(gt, pre, thresholds):
    num = gt.shape[0]
    pre_index = []
    count_low = 0
    count_up = 0
    low_time = 0
    up_time = 0
    waste_time = 0
    match_count = 0
    dic = gene_dict(order_)
    for i in range(num):
        t1 = time.time()
        low_v, up_v = find_bounds(gt[i], pre[i])
        # flag = 1
        low_time += time.time() - t1
        if low_v > thresholds[i]:
            count_low += 1
            continue
        if up_v < thresholds[i]:
            count_up += 1
            continue
        t1 = time.time()
        loss = JSD_(gt[i], pre[i])
        waste_time += time.time() - t1
        # if loss > thresholds[i]:
        #     pre_index.append(1)
        # else:
        #     pre_index.append(0)
    print('lowbound', count_low, 'upbound', count_up)
    print('match case: ', match_count)
    print(low_time, up_time, waste_time)
    return sum([low_time, up_time, waste_time])


def Bounds(gt, pre, thresholds):
    num = gt.shape[0]
    pre_index = []
    count_low = 0
    count_up = 0
    low_time = 0
    up_time = 0
    waste_time = 0
    for i in range(num):
        t1 = time.time()
        flag = L1_norm_2(gt[i], pre[i], thresholds[i])
        # flag = 1
        low_time += time.time() - t1
        if flag == 1:  # low bound larger than threshold
            pre_index.append(1)
            count_low += 1
        else:
            t1 = time.time()
            flag2 = UpperBound_order(gt[i], pre[i], thresholds[i], 3)
            up_time += time.time() - t1
            if flag2:
                count_up += 1
            else:
                t1 = time.time()
                loss = JSD_(gt[i], pre[i])
                waste_time += time.time() - t1
            # if loss > thresholds[i]:
            #     pre_index.append(1)
            # else:
            #     pre_index.append(0)
    print(count_low, count_up)
    print(low_time, up_time, waste_time)
    return sum([low_time, up_time, waste_time])


def get_f1(p, r, t):
    max = -1
    tew = 0
    for x, y, th in zip(p, r, t):
        if x != 0 or y != 0:
            val = (2 * (x * y) / (x + y))
        else:
            val = 0
        if max < val:
            max = val
            tew = th
        print(val)
    print('best threshold', tew)


# def orginial_cost():
#     root_path = 'E:\\VLDB_model\\'
#     DBNAME = ['Influencer', 'Speech', 'TED']
#     thresholds = [0.00017839953943621367, 2.3825053574455524e-07, 4.918874037684873e-05]
#     C = [0.8, 0.9, 0.9]
#     DB = 0
#     pos_f = np.load(root_path + DBNAME[DB] + '\\pos_f.npz')
#     neg_f = np.load(root_path + DBNAME[DB] + '\\neg_f.npz')
#     pos_a = np.load(root_path + DBNAME[DB] + '\\pos_a.npz')
#     neg_a = np.load(root_path + DBNAME[DB] + '\\neg_a.npz')
#     times = []
#     print(pos_f['gt'].shape)
#     print(neg_f['gt'].shape)
#     loss_1_f, t1 = self_JSD(pos_f['gt'], pos_f['pre'])
#     loss_1_a, t2 = self_MSE(pos_a['gt'], pos_a['pre'])
#     loss_2_f, t3 = self_JSD(neg_f['gt'], neg_f['pre'])
#     loss_2_a, t4 = self_MSE(neg_a['gt'], neg_a['pre'])
#     times.append(t1)
#     times.append(t2)
#     times.append(t3)
#     times.append(t4)
#     st = time.time()
#     tmp_f = (1 - C[DB]) * loss_1_f + C[DB] * loss_1_a
#     pre_pos = np.where(tmp_f < thresholds[DB], 0, 1)
#     tmp_f = (1 - C[DB]) * loss_2_f + C[DB] * loss_2_a
#     pre_neg = np.where(tmp_f < thresholds[DB], 0, 1)
#     t5 = time.time() - st
#     times.append(t5)
#     print(sum(pre_pos), sum(pre_neg))
#     print('----------------')
#     print(sum(times))
#     print(times)
#
#
# def optimized_cost():
#     root_path = 'E:\\VLDB_model\\'
#     DBNAME = ['Influencer', 'Speech', 'TED']
#     C = [0.8, 0.9, 0.9]
#     thresholds = [0.00017839953943621367, 2.3825053574455524e-07, 4.918874037684873e-05]
#     DB = 0
#     pos_f = np.load(root_path + DBNAME[DB] + '\\pos_f.npz')
#     neg_f = np.load(root_path + DBNAME[DB] + '\\neg_f.npz')
#     pos_a = np.load(root_path + DBNAME[DB] + '\\pos_a.npz')
#     neg_a = np.load(root_path + DBNAME[DB] + '\\neg_a.npz')
#     times = []
#     print(pos_f['gt'].shape, neg_f['gt'].shape)
#     loss_1_a, t2 = self_MSE(pos_a['gt'], pos_a['pre'])
#     loss_2_a, t4 = self_MSE(neg_a['gt'], neg_a['pre'])
#
#     # threholds[DB] - loss_i_a
#     times.append(t2)
#     times.append(t4)
#
#     t1 = boundaries(pos_f['gt'], pos_f['pre'], (thresholds[DB] * 2.5 - C[DB] * loss_1_a) / (1 - C[DB]))
#     t2 = boundaries(neg_f['gt'], neg_f['pre'], (thresholds[DB] * 2.5 - C[DB] * loss_2_a) / (1 - C[DB]))
#     # boundaries_upper()
#     # t1 = boundaries_upper_2(pos_f['gt'], pos_f['pre'], (thresholds[DB] * 2.5 - C[DB] * loss_1_a) / (1 - C[DB]))
#     # t2 = boundaries_upper_2(neg_f['gt'], neg_f['pre'], (thresholds[DB] * 2.5 - C[DB] * loss_2_a) / (1 - C[DB]))
#     times.append(t1)
#     times.append(t2)
#     print('----------------')
#     print(sum(times))
#     print(times)


def count_number(gt, pre, thresholds):
    num = gt.shape[0]
    pre_index = []
    count_low = 0
    count_up = 0

    low_time = 0
    up_time = 0
    waste_time = 0
    xor = 0
    for k in range(num):
        t1 = time.time()
        print('gt: ', gt[k])
        print('pre: ', pre[k])
        # flags = UpperBound_2_2(gt[k], pre[k], thresholds[k])
        # xor += (flags[0]) # ,  flags[1]
        # UpperBound_2(gt[k], pre[k], thresholds[k])
    print('total number and unmatched number: ', num, xor)


def count_seg(DB, pro_id):
    root_path = 'E:\\VLDB_model\\'
    DBNAME = ['Influencer', 'Speech', 'TED']
    C = [0.8, 0.9, 0.9]
    thresholds = [0.00017839953943621367, 8.884975977707654e-05, 4.918874037684873e-05]
    # DB = 2
    pos_f = np.load(root_path + DBNAME[DB] + '\\pos_f.npz')
    neg_f = np.load(root_path + DBNAME[DB] + '\\neg_f.npz')
    pos_a = np.load(root_path + DBNAME[DB] + '\\pos_a.npz')
    neg_a = np.load(root_path + DBNAME[DB] + '\\neg_a.npz')
    times = []
    print(pos_f['gt'].shape, neg_f['gt'].shape)
    print('th: ', thresholds[DB])
    # pro_id =-1
    loss_1_a, t2 = self_MSE(pos_a['gt'], pos_a['pre'])
    # loss_2_a, t4 = self_MSE(neg_a['gt'], neg_a['pre'])

    if pro_id == 1:
        # count_number(pos_f['gt'], pos_f['pre'], (thresholds[DB] * 1 - C[DB] * loss_1_a) / (1 - C[DB]))
        Lis = []
        length = pos_f['gt'].shape[0]
        # writer = pd.ExcelWriter('A.xlsx')
        tmp = np.hstack([pos_f['gt'], pos_f['pre']])
        print(tmp.shape)
        tmp = tmp.reshape(length * 2, 400)
        print(tmp.shape)
        # data = pd.DataFrame(tmp)
        # data.to_excel(writer, 'normal')
        np.savetxt('normal_part_1.txt', tmp, fmt='%.3e')
        length = neg_f['gt'].shape[0]
        tmp = np.hstack([neg_f['gt'], neg_f['pre']])
        print(tmp.shape)
        tmp = tmp.reshape(length * 2, 400)
        print(tmp.shape)
        np.savetxt('novel_part_1.txt', tmp, fmt='%.3e')
        # data = pd.DataFrame(tmp)
        # data.to_excel(writer, 'novel')
        # writer.save()
        # for i in range(lenght):
        #     # if i > 100:
        #     #     break
        #     if i % 10 == 0:
        #         print('handle ', i)
        #     Lis.append(pos_f['gt'][i])
        #     Lis.append(pos_f['pre'][i])

        Lis = np.array(Lis)

        print('----------------Novel--------')
        # count_number(neg_f['gt'], neg_f['pre'], (thresholds[DB] * 1 - C[DB] * loss_2_a) / (1 - C[DB]))
    return


def run_process(DB, pro_id):
    root_path = 'E:\\VLDB_model\\'
    DBNAME = ['Influencer', 'Speech', 'TED']
    C = [0.8, 0.9, 0.9]
    thresholds = [0.00017839953943621367, 8.884975977707654e-05, 4.918874037684873e-05]
    # DB = 2
    pos_f = np.load(root_path + DBNAME[DB] + '\\pos_f.npz')
    neg_f = np.load(root_path + DBNAME[DB] + '\\neg_f.npz')
    pos_a = np.load(root_path + DBNAME[DB] + '\\pos_a.npz')
    neg_a = np.load(root_path + DBNAME[DB] + '\\neg_a.npz')
    times = []
    print(pos_f['gt'].shape, neg_f['gt'].shape)
    print('th: ', thresholds[DB])
    # pro_id =-1
    t1 = 0
    t2 = 0
    t3 = 0
    t4 = 0
    num1 = pos_a['gt'].shape[0]
    list_2 = []
    if pro_id == -1:
        for i in range(num1):
            if i >1500:
                break
            t7 = JSD_analysis(pos_f['gt'][i], pos_f['pre'][i])
            list_2.append(t7)
            print(i, t7)
        list_2 = np.array(list_2)
        print(list_2.mean())
        # maxx / sum_, maxx_2 / sum_, maxx_3 / sum_, pos_max, pos_max_2, pos_max_3, ignore / sum_

    if pro_id == 0:
        loss_1_f, t1 = self_JSD(pos_f['gt'], pos_f['pre'])
        loss_1_a, t2 = self_MSE(pos_a['gt'], pos_a['pre'])
        loss_2_f, t3 = self_JSD(neg_f['gt'], neg_f['pre'])
        loss_2_a, t4 = self_MSE(neg_a['gt'], neg_a['pre'])

        tmp_f = (1 - C[DB]) * loss_1_f + C[DB] * loss_1_a
        print('pos_mean:', np.mean(tmp_f))
        tmp_f = (1 - C[DB]) * loss_2_f + C[DB] * loss_2_a
        print('neg_mean:', np.mean(tmp_f))
    if pro_id == 1:
        loss_1_a, t2 = self_MSE(pos_a['gt'], pos_a['pre'])
        loss_2_a, t4 = self_MSE(neg_a['gt'], neg_a['pre'])
        t1 = boundaries_L1(pos_f['gt'], pos_f['pre'], (thresholds[DB] * 1 - C[DB] * loss_1_a) / (1 - C[DB]))
        t3 = boundaries_L1(neg_f['gt'], neg_f['pre'], (thresholds[DB] * 1 - C[DB] * loss_2_a) / (1 - C[DB]))

    if pro_id == 2:
        loss_1_a, t2 = self_MSE(pos_a['gt'], pos_a['pre'])
        loss_2_a, t4 = self_MSE(neg_a['gt'], neg_a['pre'])
        t1 = boundaries_upper_new(pos_f['gt'], pos_f['pre'], (thresholds[DB] * 1 - C[DB] * loss_1_a) / (1 - C[DB]))
        t3 = boundaries_upper_new(neg_f['gt'], neg_f['pre'], (thresholds[DB] * 1 - C[DB] * loss_2_a) / (1 - C[DB]))
    if pro_id == 3:
        loss_1_a, t2 = self_MSE(pos_a['gt'], pos_a['pre'])
        loss_2_a, t4 = self_MSE(neg_a['gt'], neg_a['pre'])
        # t1 = Bounds(pos_f['gt'], pos_f['pre'], (thresholds[DB] * 1 - C[DB] * loss_1_a) / (1 - C[DB]))
        # t3 = Bounds(neg_f['gt'], neg_f['pre'], (thresholds[DB] * 1 - C[DB] * loss_2_a) / (1 - C[DB]))
        t1 = Bounds_new(pos_f['gt'], pos_f['pre'], (thresholds[DB] * 1 - C[DB] * loss_1_a) / (1 - C[DB]))
        t3 = Bounds_new(neg_f['gt'], neg_f['pre'], (thresholds[DB] * 1 - C[DB] * loss_2_a) / (1 - C[DB]))
    if pro_id == 4:
        loss_1_a, t2 = self_MSE(pos_a['gt'], pos_a['pre'])
        loss_2_a, t4 = self_MSE(neg_a['gt'], neg_a['pre'])
        t1 = Bounds_L(pos_f['gt'], pos_f['pre'], (thresholds[DB] * 1 - C[DB] * loss_1_a) / (1 - C[DB]))
        t3 = Bounds_L(neg_f['gt'], neg_f['pre'], (thresholds[DB] * 1 - C[DB] * loss_2_a) / (1 - C[DB]))
    # threholds[DB] - loss_i_a
    times.append(t2)
    times.append(t4)

    # t1 = boundaries(pos_f['gt'], pos_f['pre'], (thresholds[DB] * 2.5 - C[DB] * loss_1_a) / (1 - C[DB]))
    # t2 = boundaries(neg_f['gt'], neg_f['pre'], (thresholds[DB] * 2.5 - C[DB] * loss_2_a) / (1 - C[DB]))
    # boundaries_upper()
    # t1 = boundaries_upper_2(pos_f['gt'], pos_f['pre'], (thresholds[DB] * 2.5 - C[DB] * loss_1_a) / (1 - C[DB]))
    # t2 = boundaries_upper_2(neg_f['gt'], neg_f['pre'], (thresholds[DB] * 2.5 - C[DB] * loss_2_a) / (1 - C[DB]))
    times.append(t1)
    times.append(t3)
    print('AAAA----------------AAAA')
    print(sum(times))
    print(times)
    print('AAAA----------------AAAA')
    return sum(times)


# def run_process(pro_id):
#     if pro_id == 0:
#         orginial_cost()
#     if pro_id == 1:
#         lowbound_cost()
#     if pro_id == 2:
#         upbound_cost()
#     if pro_id == 3:
#         optimized_cost()
#     return

# def JSD_analysis(gt, pre):
#     maxx = -1
#     maxx_2 = -1
#     maxx_3 = -1
#     pos_max = -1
#     pos_max_2 = -1
#     pos_max_3 = -1
#     mixed = (gt + pre) / 2
#     sum_ = 0
#     ignore = 0
#     for i in range(400):
#         item = (gt[i] * np.log(gt[i] / mixed[i]) + pre[i] * np.log(pre[i] / mixed[i])) * 0.5
#         # # item = (gt[i] * np.log(gt[i] / pre[i]) + pre[i] * np.log(pre[i] / gt[i])) * 0.5
#         # # print(item)
#         # if item > maxx:
#         #     maxx_3 = maxx_2
#         #     maxx_2 = maxx
#         #     maxx = item
#         #     pos_max_3 = pos_max_2
#         #     pos_max_2 = pos_max
#         #     pos_max = i
#         # else:
#         #     if item > maxx_2:
#         #         maxx_3 = maxx_2
#         #         maxx_2 = item
#         #         pos_max_3 = pos_max_2
#         #         pos_max_2 = i
#         #     else:
#         #         if item > maxx_3:
#         #             maxx_3 = item
#         #             pos_max_3 = i
#         th = 2**(-15)
#         if gt[i] < 1e-6 or pre[i] < 1e-6:
#             # print('!!!')
#             ignore += item
#         sum_ += item
#     return maxx / sum_, maxx_2 / sum_, maxx_3 / sum_, pos_max, pos_max_2, pos_max_3, ignore / sum_


def JSD_analysis(gt, pre):
    maxx = -1
    maxx_2 = -1
    maxx_3 = -1
    pos_max = -1
    pos_max_2 = -1
    pos_max_3 = -1
    mixed = (gt + pre) / 2
    sum_ = 0
    ignore = 0
    sum_ = JSD_(gt, pre)
    th = 2 ** (-20)
    idx = np.where((gt < th) | (pre < th))
    ignor = JSD_(gt[idx], pre[idx])
    if ignor>0:
        print('!')
    return  ignore / sum_


def JSD_analysis_multi(no, gt, pre):
    maxx = -1
    maxx_2 = -1
    maxx_3 = -1
    pos_max = -1
    pos_max_2 = -1
    pos_max_3 = -1
    flag_num = pre.min() / gt.min()
    stl = str(pre.min())
    c = stl.split('e')[-1]
    d = float(c)
    e_th = np.power(10, d)
    mixed = (gt + pre) / 2
    sum_ = 0
    sum_2 = 0
    ignore = 0
    max_num = 10
    max_list = [-1] * 10
    pos_list = [-1] * 10
    ig_count = 0
    for i in range(400):
        item = (gt[i] * np.log(gt[i] / mixed[i]) + pre[i] * np.log(pre[i] / mixed[i])) * 0.5
        # item_2 = (gt[i] * np.log(gt[i] / pre[i]) + pre[i] * np.log(pre[i] / gt[i])) * 0.5
        # print(item)
        for j in range(10):
            if item > max_list[j]:
                max_list.insert(j, item)
                pos_list.insert(j, i)
                break

        if gt[i] < 1e-6 or pre[i] < 1e-6:
            # print('!!!')
            ignore += item
            ig_count += 1
        sum_ += item
        # sum_2 += item_2
    if ig_count == 0 or flag_num > 50 or flag_num < 1 / 50:
        ignore = 0
        ig_count = 0
    else:
        # print('No. ', no, ' ', ig_count, ignore / sum_, sum_, gt.max(), pre.max(), gt.min(), pre.min(), e_th)
        pass
    return max_list[:10], pos_list[:10], ignore, ig_count, sum_, flag_num


def sub_count_inpotence(gt, pre):
    length_pos = gt.shape[0]
    hist = np.zeros([400])
    hits = np.zeros([400])
    list_prop = [[1, 0]] * 400

    tar = [2, 4, 303, 304, 353, 371]
    tar_sum = np.zeros([len(tar)])
    props = []
    igs = []
    ig_cs = []
    wd = 0
    for i in range(length_pos):
        # if i % 50 == 0:
        #     print('handling ', i)
        # if i > 100:
        #     break
        # prop1, prop2, prop3, idx, idx_2, idx_3, ig = JSD_analysis(gt[i], pre[i])
        max_list, pos_list, ignore_v, ig_count, KL_value, flag_num = JSD_analysis_multi(i, gt[i], pre[i])
        if flag_num > 50:
            wd += 1
        idx = pos_list[0]
        tmp = (gt[i][idx] + pre[i][idx]) * 0.5
        hits[idx] += tmp
        for j in range(len(tar)):
            tar_sum[j] += gt[i][j] + pre[i][j]
        hist[idx] += 1
        minn, maxx = list_prop[idx]
        prop = 0
        for p in max_list:
            prop += p / KL_value
        props.append(prop)

        if ignore_v != 0:
            igs.append(ignore_v / KL_value)
            ig_cs.append(ig_count)
        if minn > prop:
            minn = prop
        if maxx < prop:
            maxx = prop
        list_prop[idx] = [minn, maxx]
    list_prop = np.array(list_prop)
    props = np.array(props)
    igs = np.array(igs)
    ig_cs = np.array(ig_cs)
    print(len(pos_list))
    print('------------')
    print('aver top-10 props: ', np.mean(props))
    print('min top-10 props: ', np.min(props))
    print('max top-10 props: ', np.max(props))
    print('less than 1e-6: ', np.mean(igs))
    print('less than 1e-6: ', np.min(igs))
    print('less than 1e-6: ', np.max(igs))
    print('wd: ', wd)
    print('less than 1e-6 count: ', np.mean(ig_cs), ig_cs.shape)
    print('less than 1e-6 count: ', np.min(ig_cs))
    print('less than 1e-6 count: ', np.max(ig_cs))
    print(list_prop.min(), list_prop.max())
    print(np.max(hist))
    scores = []
    for i in range(400):
        if hist[i] > 50:
            if hist[i] == 0:
                tmp = 0
            else:
                tmp = hits[i] / hist[i]
            scores.append(tmp)
            print(i, hist[i], tmp, list_prop[i])
    scor = np.array(scores)
    print(np.mean(scor))
    print(gt.max(), gt.min())
    print(pre.max(), pre.min())
    print('--------------')
    # for item in tar_sum:
    #     print(item / length_pos / 2)


def count_importance(DB):
    root_path = 'E:\\VLDB_model\\'
    DBNAME = ['Influencer', 'Speech', 'TED']
    C = [0.8, 0.9, 0.9]
    thresholds = [0.00017839953943621367, 8.884975977707654e-05, 4.918874037684873e-05]
    # DB = 2
    pos_f = np.load(root_path + DBNAME[DB] + '\\pos_f.npz')
    neg_f = np.load(root_path + DBNAME[DB] + '\\neg_f.npz')
    pos_a = np.load(root_path + DBNAME[DB] + '\\pos_a.npz')
    neg_a = np.load(root_path + DBNAME[DB] + '\\neg_a.npz')
    times = []
    print(pos_f['gt'].shape, neg_f['gt'].shape)
    print('th: ', thresholds[DB])

    # loss_1_a, t2 = self_MSE(pos_a['gt'], pos_a['pre'])
    # loss_2_a, t4 = self_MSE(neg_a['gt'], neg_a['pre'])
    sub_count_inpotence(pos_f['gt'], pos_f['pre'])
    sub_count_inpotence(neg_f['gt'], neg_f['pre'])


def test_dic():
    list_ = [(4, 1), (5, 1), (5, 1), (1, 2)]
    print(list_)
    hash_dic = {}
    for item in list_:
        if item in hash_dic.keys():
            hash_dic[item] += 1
        else:
            hash_dic.update({item: 1})
        # print(hash_dic)
    print(hash_dic)


def count_pair(h_1, h_2):
    pair_dict = {}
    for x, y in zip(h_1, h_2):
        if (x, y) in pair_dict.keys():
            pair_dict[(x, y)] += 1
        else:
            pair_dict.update({(x, y): 1})
    return pair_dict


def hash_multi(v1, v2):
    val = 1
    seg_number = 20  # 2**(-seg_number-1)
    step = 2
    hash_v1 = np.zeros_like(v1, dtype=int)
    hash_v2 = np.zeros_like(v2, dtype=int)
    for i in range(seg_number):
        if i == 0:
            val = 1 / step
            print('val: ', i, [val, 1], [-(i + 1), -i])
            hash_v1 = np.where((v1 >= val), i * np.ones_like(v1, dtype=int), hash_v1)
            hash_v2 = np.where((v2 >= val), i * np.ones_like(v1, dtype=int), hash_v2)
            continue
        if i == seg_number - 1:
            hash_v1 = np.where((val > v1) & (v1 >= 1e-6), i * np.ones_like(v1, dtype=int), hash_v1)
            hash_v2 = np.where((val > v2) & (v2 >= 1e-6), i * np.ones_like(v2, dtype=int), hash_v2)
            print('val: ', i, [1e-6, val], [-(i + 1), -i])
            break
        print('val: ', i, [val / step, val], [-(i + 1), -i])
        hash_v1 = np.where((val > v1) & (v1 >= val / step), i * np.ones_like(v1, dtype=int), hash_v1)
        hash_v2 = np.where((val > v2) & (v2 >= val / step), i * np.ones_like(v1, dtype=int), hash_v2)
        val = val / step
    return hash_v1, hash_v2


if __name__ == '__main__':
    path = 'D:\\LAB\\code\\ICDE\\action_recognition\\'
    # thresholds = [0.00017839953943621367, 2.3825053574455524e-07, 4.918874037684873e-05]
    # for i in thresholds:
    #     print((i*10))

    # 0 no; 1 low bound; 2 up bound; 3 low-up bound

    tnp = np.array([[1, 3], [5, 3]])
    tn2p = np.array([[1, 3], [4, 5]])
    idx = np.array([1, 0, 0, 1, 1])
    ll = np.bitwise_xor(tnp, tn2p)
    indx = [0, 2]
    db_id = 0
    order_ = 3
    # test_dic()
    # list = [0,1,2,3,4,5]
    # print(list[:2])
    # count_seg(db_id, 1)
    # count_importance(db_id)
    # last = 0.000000953674 * 2 1.907348e-06
    # print(last)
    # testv_1 = np.array([0.4, 0.5, 0.02])
    # print(hash_multi(testv_1, testv_1))
    # t1 = np.array([0, 1, 3, 1])
    # t2 = np.array([0, 1, 3, 2])
    # print(count_pair(t1, t2))

    # print(tnp.shape, tn2p)
    # print(np.hstack([tnp, tn2p]))
    # res = np.hstack([tnp, tn2p])
    # tf = res.reshape(4, 2)
    # print(tf)
    t0 = run_process(db_id, -1)
    # t1 = run_process(db_id, 1)
    # t2 = run_process(db_id, 2)
    # t3 = run_process(db_id, 3)
    # t4 = run_process(db_id, 4)
    # print(t0, t1, t2, t3, t4)

    # x = np.array([0.2, 0.8])
    # y = np.array([0.5, 0.5])
    # ans = KLD(x, y)
    # print(ans)
    # norm_1 = np.sum(np.abs(x - y))
    # ans2 = 0.5 * norm_1 * norm_1
    # print(ans2)
    # print(norm_1)
    # print(tn2p[np.where(idx == 0)])
    # print(np.where(tnp>3, 1,0))
    # print(ll)
    # print(np.where(tnp > 2, 0, 1))
    # orginial_cost()  # 17s 10s 13s
    # optimized_cost()  #
    #
    # gt_data = generate_test_data(1, 400)
    # pre_data = generate_test_data(1, 400)
    # tensor_gt = torch.softmax(tensorlize_array(gt_data), -1)
    # tensor_pre = torch.softmax(tensorlize_array(pre_data), -1)
    # pos = np.load(path + 'pos_f.npz')
    # pref = pos['pre']
    # gtf = pos['gt']
    # print(pref.shape, gtf.shape)
    # # KLD(tensor_gt[0], tensor_pre[0])
    # # KLD(tensor_pre[0], tensor_gt[0])
    # # kl = nn.KLDivLoss(reduction='sum')
    # # print(kl(tensor_gt[0].log(), tensor_pre[0]))
    # # print(kl(tensor_gt[0], tensor_pre[0].log()))
    # # crition_JS = JSD()
    # # print(crition_JS(tensor_gt[0], tensor_pre[0]))
    # # print(JSD_(tensor_gt[0], tensor_pre[0]))
    # # print(tensor_gt)
    # # print(tensor_pre)
    # arr, ind = torch.sort(tensor_gt[0], -1, True)
    # # print(arr)
    # # print(ind)
    # # no_opt_JSD(tensor_pre, tensor_gt)
    # # self_JSD(tensor_pre,tensor_gt)
    # # self_L1(tensor_pre,tensor_gt, 0.5)
    # # L1_norm(tensor_pre, tensor_gt)
    # # cProfile.run("no_opt_JSD(tensor_pre, tensor_gt)")
    # x1 = torch.tensor([1, 2, 3, 4, 5])
    # y1 = torch.tensor([1, 4, 2, 3, 5])
    #
    # # print(hash_match_1(x1, y1, 6))
    # x1 = tensor_gt[0]
    # y1 = tensor_pre[0]
    # print(x1.max(), x1.min())
    # zeros = torch.zeros(x1.shape)
    # ones = torch.ones(x1.shape)
    # x = torch.where(x1 < 0.01, zeros, ones).byte()
    # y = torch.where(y1 > 3, zeros, ones).byte()
    # # print(x, y)
    # # print(torch.bitwise_xor(x, y))
    # # print(torch.split(x,1,-1))
    # compass(x, x1)
    #
    # x = [0.1, 0.1, 0.3, 0.5, 0.6]
    # y = [1, 1, 1, 0, 0]
    # x = np.array(x)
    # y = np.array(y)
    # p, r, t = precision_recall_curve(y, x, pos_label=0)
    #
    # # f = f1_score(y, x)
    # get_f1(p, r, t)
    # print(p)
    # print(r)
    # print(t)
    #
    # f, p, t = roc_curve(y, x, pos_label=0)
    # roc_auc = auc(f, p)
    # print(p)
    # print(f)
    # print(t)
    # print(roc_auc)
    # # print(f)

# 0.008783578872680664 0.0002927859624226888
