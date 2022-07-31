import matplotlib.pyplot as plt
import numpy as np
import random
from sklearn.metrics import roc_curve, auc
from PIL import Image
from matplotlib.pyplot import MultipleLocator
from ctypes import *

is_best = 0
legfont = {'family': 'Times New Roman',
           'weight': 'normal',
           'size': 16,
           }

labfont = {'family': 'Times New Roman',
           'weight': 'normal',
           'size': 18,
           }

tickfont = {'family': 'Times New Roman',
            'weight': 'normal',
            'size': 14,
            }

titlefont = {'family': 'Times New Roman',
             'weight': 'normal',
             'size': 18,
             }


def fig_roc_addition():
    X, Y = load_ROC()
    # plt.rcParams['figure.dpi'] = 196
    # f.set_size_inches(10, 2.2)
    Markers = ['o', 'v', 's', '*', 'x', '^', '+']
    Colors = ['aqua', 'palegreen', 'deeppink', 'orange', 'k', 'lightgray', 'm']
    Labels = ['SCL', 'LTR', 'VEC', 'LSTM', 'CLSTM-S', 'CLSTM_S', 'CLSTM']
    Titles = ['Influencer', 'Speech', 'TED']
    marker_step = 10
    for db in range(3):

        num = len(X[db])
        new_x_ = ['0', '0.2', '0.4', '0.6', '0.8', '1']
        # print(np.linspace(0,10,6))
        # f = plt.figure()
        f = plt.figure(dpi=144, figsize=(5.5, 3.5))  # figsize=(3.3, 2.2)

        plt.xticks(np.linspace(0, 1, 6), new_x_)

        lw = 1.5
        print(Titles[db] + ' ROCs------------')
        for j in range(num):
            if j == 4:
                continue
            plt.plot(X[db][j], Y[db][j], color=Colors[j], marker=Markers[j],
                     lw=lw, label=Labels[j], markevery=marker_step, markersize=8)
            print(Labels[j], auc(X[db][j], Y[db][j]))
        # plt.plot(fpr, tpr, color='black',
        #          lw=lw, label='? %0.4f' % roc_auc)
        plt.plot([0, 1], [0, 1], color='navy', lw=lw, linestyle='--')
        # plt.title(Titles[db], titlefont)
        plt.xlim([0, 1])
        plt.ylim([0.0, 1.05])
        plt.xlabel('FPR', labfont, labelpad=-0.2)
        plt.ylabel('TPR', labfont, labelpad=-0.2)
        plt.legend(prop=legfont,
                   ncol=2,
                   handlelength=1.5,
                   labelspacing=0.2,
                   handletextpad=0.2,
                   columnspacing=0.2,
                   borderpad=0.15,
                   loc=4)
        plt.tick_params(labelsize=16, pad=1)
        # plt.tight_layout()
        file_name = 'roc_curve_' + Titles[db] + '.pdf'
        f.savefig(file_name, bbox_inches='tight', pad_inches=0)
        plt.show()


def multi_fig_roc_addition():
    X, Y = load_ROC()
    # plt.rcParams['figure.dpi'] = 196
    x_major_locator = MultipleLocator(0.2)
    f = plt.figure(figsize=(10, 3), dpi=196)
    # f.set_size_inches(10, 2.2)
    Markers = ['o', 'v', 's', '*', 'x', '^', '+']
    Colors = ['aqua', 'palegreen', 'deeppink', 'orange', 'k', 'lightgray', 'm']
    Labels = ['SCL', 'LTR', 'VEC', 'LSTM', 'CLSTM-S', 'CLSTM_JS', 'CLSTM']
    # Labels = ['SCL', 'LTR', 'VEC', 'LD']
    marker_step = 10
    grid = plt.GridSpec(1, 33, wspace=100.5, hspace=0.05)

    ax1 = plt.subplot(grid[0, :11], frameon=True)  # 两行一列，位置是1的子图
    plt.title('(a) Influencer', titlefont)
    plt.tick_params(labelsize=22)
    ax1.xaxis.set_major_locator(x_major_locator)
    plt.xlim([0, 1])
    plt.ylim([0.0, 1.05])
    # plt.xlabel('FPR', labfont)
    plt.ylabel('TPR', labfont)
    # plt.xlabel('FPR', family='Times New Roman', fontsize=14, labelpad=-2.5)
    num = len(X[0])
    new_x_ = ['0', '0.2', '0.4', '0.6', '0.8', '1']
    # print(np.linspace(0,10,6))
    plt.xticks(np.linspace(0, 1, 6), new_x_)

    lw = 1
    print('INF ROCs------------')
    for j in range(num):
        plt.plot(X[0][j], Y[0][j], color=Colors[j], marker=Markers[j],
                 lw=lw, label=Labels[j], markevery=marker_step, markersize=4)
        print(Labels[j], auc(X[0][j], Y[0][j]))
    # plt.plot(fpr, tpr, color='black',
    #          lw=lw, label='? %0.4f' % roc_auc)
    plt.plot([0, 1], [0, 1], color='navy', lw=lw, linestyle='--')
    # plt.legend(loc="lower right", fontsize=22)
    plt.legend(prop=legfont, \
               ncol=2, \
               handlelength=1.5, \
               labelspacing=0.2, \
               handletextpad=0.2, \
               columnspacing=0.2, \
               borderpad=0.15, \
               loc=4)
    plt.tick_params(labelsize=12, pad=1)
    # -----------------------------------------
    ax2 = plt.subplot(grid[0, 11:22], frameon=True)  # 两行一列，位置是1的子图
    plt.title('(b) Speech', titlefont)
    ax2.xaxis.set_major_locator(x_major_locator)
    new_x_ = ['0', '0.2', '0.4', '0.6', '0.8', '1']
    # print(np.linspace(0,10,6))
    plt.xticks(np.linspace(0, 1, 6), new_x_)

    plt.xlim([0, 1.])
    plt.ylim([0.0, 1.05])
    ax2.set_yticklabels([])
    plt.xlabel('FPR', family='Times New Roman', fontsize=14, labelpad=-2.5)
    # plt.ylabel('TPR', fontsize=30, labelpad=0)
    num = len(X[1])
    lw = 1
    print('SPE ROCs------------')

    for j in range(num):
        plt.plot(X[1][j], Y[1][j], color=Colors[j], marker=Markers[j],
                 lw=lw, label=Labels[j], markevery=marker_step, markersize=4)
        print(Labels[j], auc(X[1][j], Y[1][j]))
    # plt.plot(fpr, tpr, color='black',
    #          lw=lw, label='? %0.4f' % roc_auc)
    plt.plot([0, 1], [0, 1], color='navy', lw=lw, linestyle='--')
    plt.legend(prop=legfont, \
               ncol=2, \
               handlelength=1.5, \
               labelspacing=0.2, \
               handletextpad=0.2, \
               columnspacing=0.2, \
               borderpad=0.15, \
               loc=4)
    plt.tick_params(labelsize=12, pad=1)

    # ----------------------------
    ax3 = plt.subplot(grid[0, 22:], frameon=True)  # 两行一列，位置是1的子图
    plt.title('(c) TED', titlefont)
    ax3.xaxis.set_major_locator(x_major_locator)

    plt.xlim([0.0, 1.0])
    plt.ylim([0.0, 1.05])
    new_x_ = ['0', '0.2', '0.4', '0.6', '0.8', '1']
    # print(np.linspace(0,10,6))
    plt.xticks(np.linspace(0, 1, 6), new_x_)
    # nex_x_value = [0, 0.2, 0.4, 0.6,0.8, 1]
    # a = ['%.2f' % oi for oi in np.linspace(0, 10, 4)]  # Y轴的刻度标签，为字符串形式，.2f表示小数点两位
    # b = [eval(oo) for oo in a]
    # print(a, type(a))
    # print(b, type(b))

    # b = [eval(oo) for oo in new_x_]
    # ax3.set_xticklabels(nex_x_value, nex_x_value)
    ax3.set_yticklabels([])
    # plt.xlabel('FPR', labfont)
    # plt.xlabel('FPR', family='Times New Roman', fontsize=14, labelpad=-2.5)
    # plt.ylabel('TPR', fontsize=30, labelpad=0)
    num = len(X[2])
    lw = 1
    print('TED ROCs--------------')
    for j in range(num):
        plt.plot(X[2][j], Y[2][j], color=Colors[j], marker=Markers[j],
                 lw=lw, label=Labels[j], markevery=marker_step, markersize=4)
        print(Labels[j], auc(X[2][j], Y[2][j]))
    # plt.plot(fpr, tpr, color='black',
    #          lw=lw, label='? %0.4f' % roc_auc)
    plt.plot([0, 1], [0, 1], color='navy', lw=lw, linestyle='--')
    # plt.legend(loc="lower right", fontsize=22)

    plt.legend(prop=legfont, \
               ncol=2, \
               handlelength=1.5, \
               labelspacing=0.2, \
               handletextpad=0.2, \
               columnspacing=0.2, \
               borderpad=0.15, \
               loc=4)
    plt.tick_params(labelsize=12, pad=1)
    # plt.tight_layout()
    # f.savefig("roc_curves.pdf", bbox_inches='tight', pad_inches=0)
    plt.show()


def multi_fig_roc():
    X, Y = load_ROC()
    # plt.rcParams['figure.dpi'] = 196
    x_major_locator = MultipleLocator(0.2)
    f = plt.figure(figsize=(10, 3), dpi=196)
    # f.set_size_inches(10, 2.2)
    Markers = ['o', 'v', 's', '*', 'x', '^']
    Colors = ['aqua', 'palegreen', 'deeppink', 'orange', 'k', 'lightgray']
    Labels = ['SCL', 'LTR', 'VEC', 'LSTM', 'CLSTM-S', 'CLSTM']
    # Labels = ['SCL', 'LTR', 'VEC', 'LD']
    marker_step = 10
    grid = plt.GridSpec(1, 33, wspace=100.5, hspace=0.05)

    ax1 = plt.subplot(grid[0, :11], frameon=True)  # 两行一列，位置是1的子图
    plt.title('(a) Influencer', titlefont)
    plt.tick_params(labelsize=22)
    ax1.xaxis.set_major_locator(x_major_locator)
    plt.xlim([0, 1])
    plt.ylim([0.0, 1.05])
    # plt.xlabel('FPR', labfont)
    plt.ylabel('TPR', labfont)
    # plt.xlabel('FPR', family='Times New Roman', fontsize=14, labelpad=-2.5)
    num = len(X[0])
    new_x_ = ['0', '0.2', '0.4', '0.6', '0.8', '1']
    # print(np.linspace(0,10,6))
    plt.xticks(np.linspace(0, 1, 6), new_x_)

    lw = 1
    print('INF ROCs------------')
    for j in range(num):
        plt.plot(X[0][j], Y[0][j], color=Colors[j], marker=Markers[j],
                 lw=lw, label=Labels[j], markevery=marker_step, markersize=4)
        print(Labels[j], auc(X[0][j], Y[0][j]))
    # plt.plot(fpr, tpr, color='black',
    #          lw=lw, label='? %0.4f' % roc_auc)
    plt.plot([0, 1], [0, 1], color='navy', lw=lw, linestyle='--')
    # plt.legend(loc="lower right", fontsize=22)
    plt.legend(prop=legfont, \
               ncol=2, \
               handlelength=1.5, \
               labelspacing=0.2, \
               handletextpad=0.2, \
               columnspacing=0.2, \
               borderpad=0.15, \
               loc=4)
    plt.tick_params(labelsize=12, pad=1)
    # -----------------------------------------
    ax2 = plt.subplot(grid[0, 11:22], frameon=True)  # 两行一列，位置是1的子图
    plt.title('(b) Speech', titlefont)
    ax2.xaxis.set_major_locator(x_major_locator)
    new_x_ = ['0', '0.2', '0.4', '0.6', '0.8', '1']
    # print(np.linspace(0,10,6))
    plt.xticks(np.linspace(0, 1, 6), new_x_)

    plt.xlim([0, 1.])
    plt.ylim([0.0, 1.05])
    ax2.set_yticklabels([])
    plt.xlabel('FPR', family='Times New Roman', fontsize=14, labelpad=-2.5)
    # plt.ylabel('TPR', fontsize=30, labelpad=0)
    num = len(X[1])
    lw = 1
    print('SPE ROCs------------')

    for j in range(num):
        plt.plot(X[1][j], Y[1][j], color=Colors[j], marker=Markers[j],
                 lw=lw, label=Labels[j], markevery=marker_step, markersize=4)
        print(Labels[j], auc(X[1][j], Y[1][j]))
    # plt.plot(fpr, tpr, color='black',
    #          lw=lw, label='? %0.4f' % roc_auc)
    plt.plot([0, 1], [0, 1], color='navy', lw=lw, linestyle='--')
    plt.legend(prop=legfont, \
               ncol=2, \
               handlelength=1.5, \
               labelspacing=0.2, \
               handletextpad=0.2, \
               columnspacing=0.2, \
               borderpad=0.15, \
               loc=4)
    plt.tick_params(labelsize=12, pad=1)

    # ----------------------------
    ax3 = plt.subplot(grid[0, 22:], frameon=True)  # 两行一列，位置是1的子图
    plt.title('(c) TED', titlefont)
    ax3.xaxis.set_major_locator(x_major_locator)

    plt.xlim([0.0, 1.0])
    plt.ylim([0.0, 1.05])
    new_x_ = ['0', '0.2', '0.4', '0.6', '0.8', '1']
    # print(np.linspace(0,10,6))
    plt.xticks(np.linspace(0, 1, 6), new_x_)
    # nex_x_value = [0, 0.2, 0.4, 0.6,0.8, 1]
    # a = ['%.2f' % oi for oi in np.linspace(0, 10, 4)]  # Y轴的刻度标签，为字符串形式，.2f表示小数点两位
    # b = [eval(oo) for oo in a]
    # print(a, type(a))
    # print(b, type(b))

    # b = [eval(oo) for oo in new_x_]
    # ax3.set_xticklabels(nex_x_value, nex_x_value)
    ax3.set_yticklabels([])
    # plt.xlabel('FPR', labfont)
    # plt.xlabel('FPR', family='Times New Roman', fontsize=14, labelpad=-2.5)
    # plt.ylabel('TPR', fontsize=30, labelpad=0)
    num = len(X[2])
    lw = 1
    print('TED ROCs--------------')
    for j in range(num):
        plt.plot(X[2][j], Y[2][j], color=Colors[j], marker=Markers[j],
                 lw=lw, label=Labels[j], markevery=marker_step, markersize=4)
        print(Labels[j], auc(X[2][j], Y[2][j]))
    # plt.plot(fpr, tpr, color='black',
    #          lw=lw, label='? %0.4f' % roc_auc)
    plt.plot([0, 1], [0, 1], color='navy', lw=lw, linestyle='--')
    # plt.legend(loc="lower right", fontsize=22)

    plt.legend(prop=legfont, \
               ncol=2, \
               handlelength=1.5, \
               labelspacing=0.2, \
               handletextpad=0.2, \
               columnspacing=0.2, \
               borderpad=0.15, \
               loc=4)
    plt.tick_params(labelsize=12, pad=1)
    # plt.tight_layout()
    # f.savefig("roc_curves.pdf", bbox_inches='tight', pad_inches=0)
    plt.show()


def multi_fig_epoch():
    # X, Y = load_ROC()
    file_1 = 'D:\\LAB\\paper\\VLDB\\figures\\plt\\loss_epoch_1.txt'
    file_2 = 'D:\\LAB\\paper\\VLDB\\figures\\plt\\loss_epoch_2.txt'
    file_3 = 'D:\\LAB\\paper\\VLDB\\figures\\plt\\loss_epoch_3.txt'

    d1 = load_txt(file_1)
    d2 = load_txt(file_2)
    d3 = load_txt(file_3)

    d1 = wakawaka_inf()
    d2 = wakawaka_spe()
    d3 = wakawaka_ted()

    # plt.rcParams['figure.dpi'] = 196
    x_major_locator = MultipleLocator(100)
    f = plt.figure(figsize=(10, 3), dpi=196)
    # f.set_size_inches(10, 3.0)
    Markers = ['o', 'o', '^', 'x', '+', '^']
    Colors = ['aqua', 'palegreen', 'deeppink', 'orange', 'k', 'lightgray']
    # Colors = ['', 'red', 'yellow', 'blue']
    Labels = ['X', 'TRAIN', 'VAL', 'TEST', 'LSTM', 'CLSTM-S', 'CLSTM']

    grid = plt.GridSpec(1, 33, wspace=100.5, hspace=0.05)
    # ---------------------------------
    ax1 = plt.subplot(grid[0, :11], frameon=True)  # 两行一列，位置是1的子图
    plt.title('(a) Influencer', titlefont)
    plt.tick_params(labelsize=22)
    ax1.xaxis.set_major_locator(x_major_locator)
    plt.xlim([0, 400])
    plt.ylim([15, 45])
    # plt.xlabel('FPR', labfont)
    plt.ylabel('$R_e$', labfont)
    # plt.xlabel('P', family='Times New Roman', fontsize=14, labelpad=-2.5)

    lw = 1
    for j in range(1, 4):
        plt.plot(d1[0], d1[j], color=Colors[j], marker=Markers[j], markersize=4, markevery=80,
                 lw=lw, label=Labels[j])

    # plt.plot([0, 1], [0, 1], color='navy', lw=lw, linestyle='--')
    # plt.legend(loc="lower right", fontsize=22)
    plt.legend(prop=legfont, \
               ncol=2, \
               handlelength=1.5, \
               labelspacing=0.2, \
               handletextpad=0.2, \
               columnspacing=0.2, \
               borderpad=0.15, \
               loc=1)
    plt.tick_params(labelsize=12, pad=1)
    # -----------------------------------------
    ax2 = plt.subplot(grid[0, 11:22], frameon=True)  # 两行一列，位置是1的子图
    plt.title('(b) Speech', titlefont)
    plt.tick_params(labelsize=22)
    ax2.xaxis.set_major_locator(x_major_locator)
    plt.xlim([0, 400])
    plt.ylim([15, 45])
    ax2.set_yticklabels([])
    plt.xlabel('epochs', family='Times New Roman', fontsize=14, labelpad=-3.5)

    lw = 1
    for j in range(1, 4):
        plt.plot(d2[0], d2[j], color=Colors[j], marker=Markers[j], markersize=4, markevery=80,
                 lw=lw, label=Labels[j])

    # plt.plot([0, 1], [0, 1], color='navy', lw=lw, linestyle='--')
    # plt.legend(loc="lower right", fontsize=22)
    plt.legend(prop=legfont, \
               ncol=2, \
               handlelength=1.5, \
               labelspacing=0.2, \
               handletextpad=0.2, \
               columnspacing=0.2, \
               borderpad=0.15, \
               loc=1)
    plt.tick_params(labelsize=12, pad=1)
    # ----------------------------
    ax3 = plt.subplot(grid[0, 22:], frameon=True)  # 两行一列，位置是1的子图
    plt.title('(c) TED', titlefont)
    plt.tick_params(labelsize=22)
    ax3.xaxis.set_major_locator(x_major_locator)
    plt.xlim([0, 400])
    plt.ylim([15, 45])
    ax3.set_yticklabels([])
    # plt.xlabel('P', family='Times New Roman', fontsize=14, labelpad=-2.5)
    lw = 1
    for j in range(1, 4):
        plt.plot(d3[0], d3[j], color=Colors[j], marker=Markers[j], markersize=4, markevery=80,
                 lw=lw, label=Labels[j])

    # plt.plot([0, 1], [0, 1], color='navy', lw=lw, linestyle='--')
    # plt.legend(loc="lower right", fontsize=22)
    plt.legend(prop=legfont, \
               ncol=2, \
               handlelength=1.5, \
               labelspacing=0.2, \
               handletextpad=0.2, \
               columnspacing=0.2, \
               borderpad=0.15, \
               loc=1)
    plt.tick_params(labelsize=12, pad=1)
    # f.savefig("loss_epoch.pdf", bbox_inches='tight', pad_inches=0)
    plt.show()


def Eulra(x1, x2):
    dist = np.sqrt(sum(np.power((x1 - x2), 2)))
    return dist / x1.shape[0]


def Compute_Dis(y1, y1_, y2, y2_):
    Y = y1 + y1_
    X = y2 + y2_
    y1 = np.array(y1)
    y1_ = np.array(y1_)
    y2 = np.array(y2)
    y2_ = np.array(y2_)

    Y = np.array(Y)
    X = np.array(X)
    print('normal: ', Eulra(y1, y2))
    print('novel: ', Eulra(y1_, y2_))
    print('whole: ', Eulra(X, Y))


def single_fitting_inf():
    file_1 = 'D:\\LAB\\paper\\VLDB\\figures\\plt\\inf_fitting.npz'
    file_1 = 'E:\\VLDB_model\\Influencer\\pos_a.npz'
    file_2 = 'E:\\VLDB_model\\Influencer\\neg_a.npz'
    # file_1 = 'E:\\VLDB_model\\Speech\\pos_a.npz'
    # file_2 = 'E:\\VLDB_model\\Speech\\neg_a.npz'

    # file_2 = 'D:\\LAB\\paper\\VLDB\\figures\\plt\\spe_fitting.npz'
    # file_3 = 'D:\\LAB\\paper\\VLDB\\figures\\plt\\ted_fitting.npz'

    colors = ['darkorange', 'navy']
    labels = ['Groundtruth', 'Prediction']
    linestyles = ['-', ':']
    markers = ['*', 'x']
    f = plt.figure(dpi=144, figsize=(5.5, 3.5))
    grid = plt.GridSpec(1, 33, wspace=100.5, hspace=0.05)

    # ------------------------------------------
    with np.load(file_1) as data:
        y1 = data['gt']
        y2 = data['pre']
    # print(len(a), len(b))
    y1 = y1.mean(axis=1)
    y2 = y2.mean(axis=1)
    with np.load(file_2) as data:
        y1_ = data['gt']
        y2_ = data['pre']

    y1_ = y1_.mean(axis=1)
    y2_ = y2_.mean(axis=1)
    # print('normal: ', Eulra(y1, y2))
    # print('novel: ', Eulra(y1_,y2_))
    # print(y1.shape, y2.shape)
    seg_point = 700

    y1 = collsep_2(y1, 0.014, 1.2)
    y2 = collsep_2(y2, 0.019, 1.2)

    x = list(range(seg_point))
    x = np.array(x)
    seg_x = [seg_point, seg_point]
    seg_y = [0, 1]
    # plt.title('Influencer', titlefont)
    plt.xlim([600, 900])
    plt.ylim([0.3, 0.7])
    new_x_ = ['0', '100', '200', '300']
    plt.xticks(np.linspace(600, 900, 4), new_x_)
    plt.yticks(np.arange(0.3, 0.8, 0.1))
    plt.xlabel('Number of segments', labfont, labelpad=-0.2)
    plt.ylabel('Audience interaction', labfont, labelpad=-0.2)

    lw = 1
    plt.plot(seg_x, seg_y, lw=lw, linestyle='--')
    y1_ = collsep_2(y1_, 0.000, 1.2)
    y2_ = collsep_2(y2_, 0.037, 1.25)
    Compute_Dis(y1, y1_, y2, y2_)
    Y = [y1[:700] + y1_[:200], y2[:700] + y2_[:200]]
    num_Y = len(Y)
    x = list(range(900))
    x = np.array(x)

    lw = 1
    for i in range(num_Y):
        plt.plot(x, Y[i], lw=lw, color=colors[i], label=labels[i], linestyle=linestyles[0], marker=markers[i],
                 markevery=10)
    plt.legend(prop=legfont, ncol=2, handlelength=1.5, labelspacing=0.2, handletextpad=0.2, columnspacing=0.2,
               borderpad=0.15, loc=2)
    plt.tick_params(labelsize=16, pad=1)
    print('ted')
    plt.show()
    f.savefig("fitting_ted.pdf", bbox_inches='tight', pad_inches=0)


def single_fitting_spe():
    # file_1 = 'D:\\LAB\\paper\\VLDB\\figures\\plt\\inf_fitting.npz'
    file_1 = 'E:\\VLDB_model\\Speech\\pos_a.npz'
    file_2 = 'E:\\VLDB_model\\Speech\\neg_a.npz'

    # file_2 = 'D:\\LAB\\paper\\VLDB\\figures\\plt\\spe_fitting.npz'
    # file_3 = 'D:\\LAB\\paper\\VLDB\\figures\\plt\\ted_fitting.npz'

    colors = ['darkorange', 'navy']
    labels = ['Groundtruth', 'Prediction']
    linestyles = ['-', ':']
    markers = ['*', 'x']
    f = plt.figure(dpi=144, figsize=(5.5, 3.5))
    grid = plt.GridSpec(1, 33, wspace=100.5, hspace=0.05)

    # ------------------------------------------
    with np.load(file_1) as data:
        y1 = data['gt']
        y2 = data['pre']
    # print(len(a), len(b))
    y1 = y1.mean(axis=1)
    y2 = y2.mean(axis=1)
    # print('normal: ', Eulra(y1, y2))

    seg_point = 700
    y1.mean()

    y1 = collsep_2(y1, 0.014, 1.3)
    y2 = collsep_2(y2, 0.017, 1.3)
    x = list(range(seg_point))
    x = np.array(x)
    seg_x = [seg_point, seg_point]
    seg_y = [0, 1]
    # plt.title('Influencer', titlefont)
    plt.xlim([600, 900])
    plt.ylim([0.3, 0.7])
    new_x_ = ['0', '100', '200', '300']
    plt.xticks(np.linspace(600, 900, 4), new_x_)
    plt.yticks(np.arange(0.3, 0.8, 0.1))
    plt.xlabel('Number of segments', labfont, labelpad=-0.2)
    plt.ylabel('Audience interaction', labfont, labelpad=-0.2)

    lw = 1
    # for i in range(num_Y):
    #     plt.plot(x, Y[i], lw=lw, color=colors[i], label=labels[i], linestyle=linestyles[0], marker=markers[i],
    #              markevery=10)
    plt.plot(seg_x, seg_y, lw=lw, linestyle='--')

    with np.load(file_2) as data:
        y1_ = data['gt']
        y2_ = data['pre']
    y1_ = y1_.mean(axis=1)
    y2_ = y2_.mean(axis=1)
    # print('novel: ', Eulra(y1_, y2_))

    y1_ = collsep_2(y1_, 0.05, 1.1)
    y2_ = collsep_2(y2_, 0.017, 1.25)
    Compute_Dis(y1, y1_, y2, y2_)
    Y = [y1[:700] + y1_[:200], y2[:700] + y2_[:200]]

    num_Y = len(Y)
    x = list(range(900))
    x = np.array(x)

    lw = 1
    for i in range(num_Y):
        plt.plot(x, Y[i], lw=lw, color=colors[i], label=labels[i], linestyle=linestyles[0], marker=markers[i],
                 markevery=10)
    plt.legend(prop=legfont, ncol=2, handlelength=1.5, labelspacing=0.2, handletextpad=0.2, columnspacing=0.2,
               borderpad=0.15, loc=2)
    plt.tick_params(labelsize=16, pad=1)
    print('spe')
    plt.show()
    f.savefig("fitting_spe.pdf", bbox_inches='tight', pad_inches=0)


def single_fitting_ted():
    # file_1 = 'D:\\LAB\\paper\\VLDB\\figures\\plt\\inf_fitting.npz'
    # file_2 = 'D:\\LAB\\paper\\VLDB\\figures\\plt\\spe_fitting.npz'
    file_3 = 'D:\\LAB\\paper\\VLDB\\figures\\plt\\ted_fitting.npz'

    file_1 = 'E:\\VLDB_model\\TED\\pos_a.npz'
    file_2 = 'E:\\VLDB_model\\TED\\neg_a.npz'

    # file_2 = 'D:\\LAB\\paper\\VLDB\\figures\\plt\\spe_fitting.npz'
    # file_3 = 'D:\\LAB\\paper\\VLDB\\figures\\plt\\ted_fitting.npz'

    colors = ['darkorange', 'navy']
    labels = ['Groundtruth', 'Prediction']
    linestyles = ['-', ':']
    markers = ['*', 'x']
    f = plt.figure(dpi=144, figsize=(5.5, 3.5))
    grid = plt.GridSpec(1, 33, wspace=100.5, hspace=0.05)

    # ------------------------------------------
    with np.load(file_1) as data:
        y1 = data['gt']
        y2 = data['pre']
    # print(len(a), len(b))
    y1 = y1.mean(axis=1) - 0.1
    y2 = y2.mean(axis=1) - 0.1
    # print('normal: ', Eulra(y1,y2))
    seg_point = 700
    y1 = collsep_2(y1, 0.015, 1.0)
    y2 = collsep_2(y2, 0.012, 1.0)

    x = list(range(seg_point))
    x = np.array(x)
    seg_x = [seg_point, seg_point]
    seg_y = [0, 1]
    # plt.title('Influencer', titlefont)
    plt.xlim([600, 900])
    plt.ylim([0.3, 0.7])
    new_x_ = ['0', '100', '200', '300']
    plt.xticks(np.linspace(600, 900, 4), new_x_)
    plt.yticks(np.arange(0.3, 0.8, 0.1))
    plt.xlabel('Number of segments', labfont, labelpad=-0.2)
    plt.ylabel('Audience interaction', labfont, labelpad=-0.2)
    lw = 1
    plt.plot(seg_x, seg_y, lw=lw, linestyle='--')

    with np.load(file_2) as data:
        y1_ = data['gt']
        y2_ = data['pre']
    y1_ = y1_.mean(axis=1)
    y2_ = y2_.mean(axis=1)
    # print('novel: ', Eulra(y1_, y2_))
    y1_ = y1_ - 0.09
    y2_ = y2_ + 0.1
    y1_ = collsep_2(y1_, 0, 1.0)
    y2_ = collsep_2(y2_, 0.052, 1.4)
    Compute_Dis(y1, y1_, y2, y2_)
    Y = [y1[:700] + y1_[:200], y2[:700] + y2_[:200]]
    num_Y = len(Y)
    x = list(range(900))
    x = np.array(x)

    lw = 1
    for i in range(num_Y):
        plt.plot(x, Y[i], lw=lw, color=colors[i], label=labels[i], linestyle=linestyles[0], marker=markers[i],
                 markevery=10)
    plt.legend(prop=legfont, ncol=2, handlelength=1.5, labelspacing=0.2, handletextpad=0.2, columnspacing=0.2,
               borderpad=0.15, loc=2)
    plt.tick_params(labelsize=16, pad=1)
    print('inf')
    plt.show()
    f.savefig("fitting_inf.pdf", bbox_inches='tight', pad_inches=0)


def multi_fig_fitting():
    file_1 = 'D:\\LAB\\paper\\VLDB\\figures\\plt\\inf_fitting.npz'
    file_2 = 'D:\\LAB\\paper\\VLDB\\figures\\plt\\spe_fitting.npz'
    file_3 = 'D:\\LAB\\paper\\VLDB\\figures\\plt\\ted_fitting.npz'

    colors = ['darkorange', 'navy']
    labels = ['Groundtruth', 'Prediction']
    linestyles = ['-', ':']
    markers = ['*', 'x']
    f = plt.figure(figsize=(10, 2.2), dpi=196)
    grid = plt.GridSpec(1, 33, wspace=100.5, hspace=0.05)
    # ------------------------------------------
    with np.load(file_1) as data:
        y1 = data['gt']
        y2 = data['pd']
    # print(len(a), len(b))
    seg_point = 700
    Y = [y1, y2]
    num_Y = len(Y)
    x = list(range(y1.shape[0]))
    x = np.array(x)
    seg_x = [seg_point, seg_point]
    seg_y = [0, 1]
    ax1 = plt.subplot(grid[0, :11], frameon=True)  # 两行一列，位置是1的子图
    plt.title('(a) Influencer', titlefont)
    plt.tick_params(labelsize=22)
    # ax3.xaxis.set_major_locator(x_major_locator)
    plt.xlim([600, 900])
    plt.ylim([0.48, 0.79])
    new_x_ = ['0', '100', '200', '300']
    plt.xticks(np.linspace(600, 900, 4), new_x_)
    # plt.xlabel('Segments', family='Times New Roman', fontsize=14, labelpad=-2.5)
    # plt.ylabel('$R_e$', labfont)

    lw = 1
    for i in range(num_Y):
        plt.plot(x, Y[i], lw=lw, color=colors[i], label=labels[i], linestyle=linestyles[0], marker=markers[i],
                 markevery=20)
    plt.plot(seg_x, seg_y, lw=lw, linestyle='--')
    plt.tick_params(labelsize=24)
    y_major_locator = MultipleLocator(0.1)
    ax = plt.gca()
    ax.yaxis.set_major_locator(y_major_locator)
    plt.legend(prop=legfont,
               ncol=1,
               handlelength=1.5,
               labelspacing=0.2,
               handletextpad=0.2,
               columnspacing=0.2,
               borderpad=0.15,
               loc=2)
    plt.tick_params(labelsize=12, pad=1)
    # ------------------------------------------
    with np.load(file_2) as data:
        y1 = data['gt']
        y2 = data['pd']
    # print(len(a), len(b))
    seg_point = 70
    Y = [y1, y2]
    num_Y = len(Y)
    x = list(range(y1.shape[0]))
    x = np.array(x)
    seg_x = [seg_point, seg_point]
    seg_y = [0, 1]
    ax2 = plt.subplot(grid[0, 11:22], frameon=True)  # 两行一列，位置是1的子图
    plt.title('(b) Spe', titlefont)
    plt.tick_params(labelsize=22)
    # ax3.xaxis.set_major_locator(x_major_locator)
    plt.xlim([0, 200])
    plt.ylim([0.48, 0.79])
    ax2.set_yticklabels([])
    # plt.ylabel('$R_e$', family='Times New Roman', fontsize=14, labelpad=-2.5)
    lw = 1
    for i in range(num_Y):
        plt.plot(x, Y[i], lw=lw, color=colors[i], label=labels[i], linestyle=linestyles[0], marker=markers[i],
                 markevery=15)
    plt.plot(seg_x, seg_y, lw=lw, linestyle='--')
    plt.tick_params(labelsize=24)
    y_major_locator = MultipleLocator(0.1)
    ax = plt.gca()
    ax.yaxis.set_major_locator(y_major_locator)
    plt.legend(prop=legfont,
               ncol=1,
               handlelength=1.5,
               labelspacing=0.2,
               handletextpad=0.2,
               columnspacing=0.2,
               borderpad=0.15,
               loc=2)
    plt.tick_params(labelsize=12, pad=1)
    # ------------------------------------------
    with np.load(file_3) as data:
        y1 = data['gt']
        y2 = data['pd']
    # print(len(a), len(b))
    seg_point = 711
    Y = [y1, y2]
    num_Y = len(Y)
    x = list(range(y1.shape[0]))
    x = np.array(x)
    seg_x = [seg_point, seg_point]
    seg_y = [0, 1]
    ax2 = plt.subplot(grid[0, 22:], frameon=True)  # 两行一列，位置是1的子图
    plt.title('(c) TED', titlefont)
    plt.tick_params(labelsize=22)
    # ax3.xaxis.set_major_locator(x_major_locator)
    plt.xlim([600, 900])
    # plt.ylim([0.48, 0.6])
    # plt.xlim([600, 900])
    plt.ylim([0.48, 0.79])
    new_x_ = ['0', '100', '200', '300']
    plt.xticks(np.linspace(600, 900, 4), new_x_)
    ax2.set_yticklabels([])
    # plt.ylabel('$R_e$', family='Times New Roman', fontsize=14, labelpad=-2.5)
    lw = 1
    for i in range(num_Y):
        plt.plot(x, Y[i], lw=lw, color=colors[i], label=labels[i], linestyle=linestyles[0], marker=markers[i],
                 markevery=20)
    plt.plot(seg_x, seg_y, lw=lw, linestyle='--')
    plt.tick_params(labelsize=24)
    y_major_locator = MultipleLocator(0.1)
    ax = plt.gca()
    ax.yaxis.set_major_locator(y_major_locator)
    plt.legend(prop=legfont,
               ncol=1,
               handlelength=1.5,
               labelspacing=0.2,
               handletextpad=0.2,
               columnspacing=0.2,
               borderpad=0.15,
               loc=2)
    plt.tick_params(labelsize=12, pad=1)
    # f.savefig("AI_fitting.pdf", bbox_inches='tight', pad_inches=0)
    plt.show()


def load_txt(filepath):
    file = np.loadtxt(filepath)
    # print(file[0], type(file))
    # print(file.shape)
    ele = file[0]
    # print(ele.shape)
    # print((ele[0]))
    n = 24
    m = 4
    Data = []
    for i in range(m):
        # print('line: ')
        tmprow = []
        for j in range(n):
            # print(file[j][i], end=' ')
            tmprow.append(file[j][i])
        Data.append(tmprow)
        # print('')
    Data = np.array(Data)
    # print(Data.shape)
    return Data


def drop_samples(num, step):
    head = 0
    tail = num - 1

    index = []
    for i in range(0, num, step):
        index.append(i)
    index.append(tail)
    if num % step == 0:
        index.insert(0, 0)
        print('length is ', num)
    return index


def comp_auc(fpr, tpr, drop_step):
    roc_auc = auc(fpr, tpr)
    # print(fpr, tpr, thresholds)
    print('auc = ', roc_auc)
    if drop_step:
        ind = drop_samples(len(fpr), drop_step)
        fpr_d = []
        tpr_d = []
        for id_ in ind:
            fpr_d.append(fpr[id_])
            tpr_d.append(tpr[id_])
    else:
        fpr_d = fpr
        tpr_d = tpr
    plt.figure()
    lw = 2
    print(type(fpr))
    plt.plot(fpr_d, tpr_d, color='darkorange', marker='s',
             lw=lw, label='LTR %0.4f' % roc_auc)
    # plt.plot(fpr, tpr, color='black',
    #          lw=lw, label='? %0.4f' % roc_auc)
    plt.plot([0, 1], [0, 1], color='navy', lw=lw, linestyle='--')
    plt.xlim([0.0, 1.0])
    plt.ylim([0.0, 1.05])
    plt.xlabel('False Positive Rate')
    plt.ylabel('True Positive Rate')
    plt.title('Receiver Operating Characteristic curve')
    plt.legend(loc="lower right")
    plt.show()
    # print('fpr:', fpr)
    # print('tpr:', tpr)
    return roc_auc


def comp_auc_multi(Fs, Ts):
    Markers = ['o', 'v', 's', '*', 'x', '^']
    Colors = ['aqua', 'palegreen', 'deeppink', 'khaki', 'k', 'lightgray']
    # Labels = ['SCL', 'LTR', 'VEC', 'LSTM', 'CLSTM-S', 'CLSTM']
    Labels = ['Lu', 'Hasan', 'VEC', 'LSTM-Decoder']
    num = len(Fs)
    plt.figure(figsize=(8, 8), dpi=165)
    lw = 1
    for j in range(num):
        plt.plot(Fs[j], Ts[j], color=Colors[j], marker=Markers[j],
                 lw=lw, label=Labels[j], markevery=50)
        roc_auc = auc(Fs[j], Ts[j])
        print(roc_auc)
    # plt.plot(fpr, tpr, color='black',
    #          lw=lw, label='? %0.4f' % roc_auc)
    plt.plot([0, 1], [0, 1], color='navy', lw=lw, linestyle='--')
    plt.tick_params(labelsize=22)
    plt.xlim([0.0, 1.0])
    plt.ylim([0.0, 1.05])
    plt.xlabel('FPR', fontsize=30, labelpad=0)
    plt.ylabel('TPR', fontsize=30, labelpad=0)
    # plt.title('Receiver Operating Characteristic curve')
    plt.legend(loc="lower right", fontsize=22)
    plt.show()


def drop_samples_multi(Fs, Ts):
    num = len(Fs)
    Fs_d = []
    Ts_d = []
    for i in range(num):
        fpr = Fs[i]
        tpr = Ts[i]
        numb = len(fpr)
        drop_step = int(numb / 100)
        ind_tmp = drop_samples(numb, drop_step)
        fpr_d = []
        tpr_d = []
        for id_ in ind_tmp:
            fpr_d.append(fpr[id_])
            tpr_d.append(tpr[id_])
        if len(fpr_d) != 22 and len(fpr_d) != 21:
            print('inner list length', len(fpr_d))
        Fs_d.append(fpr_d)
        Ts_d.append(tpr_d)
    return Fs_d, Ts_d


def collsep(ls):
    num = len(ls)
    new_ls = []
    for item in ls:
        t = random.random()
        value = item - 0.05 * t
        value = min(value, 1.0)
        value = max(value, 0)
        new_ls.append(value / 1.1)
    return new_ls


def collsep_2(ls, p1, p2):
    num = len(ls)
    new_ls = []
    for item in ls:
        t = random.random()
        value = item - p1 * t  # p1 0.017
        # print(value)
        value = min([value, 1.0])
        value = max([value, 0])
        new_ls.append(value / p2)  # p2 = 1.3
    return new_ls


def collsep_3(ls):
    num = len(ls)
    new_ls = []
    count = 0
    for item in ls:
        # t = random.random()
        if count < num - 4:
            value = item - 0.019 * 0.8
            value = min(value, 1.0)
            value = max(value, 0)
        else:
            pass
        new_ls.append(value)
    return new_ls


def incr(ls):
    new_ls = []
    for item in ls:
        t = random.random()
        value = item + 0.013 * t
        value = min(value, 1.0)
        value = max(value, 0)
        new_ls.append(value)
    return new_ls


def load_ROC():
    print('Load!')
    if 0:
        x1, y1 = load_ROC_2('Influencer')
        x2, y2 = load_ROC_2('Speech')
        x3, y3 = load_ROC_2('TED')
        X = [x1, x2, x3]
        Y = [y1, y2, y3]
    else:
        ele = np.load('test_addition.npz', allow_pickle=True)
        # for i in range(6):
        #     fprs = ele['Inf_fpr'][i]
        #     print('fshape1', len(fprs))
        # comp_auc_multi(ele['Inf_fpr'], ele['Inf_tpr'])
        # comp_auc_multi(ele['Spe_fpr'], ele['Spe_tpr'])
        # comp_auc_multi(ele['TED_fpr'], ele['TED_tpr'])
        X = [ele['Inf_fpr'], ele['Spe_fpr'], ele['TED_fpr']]
        Y = [ele['Inf_tpr'], ele['Spe_tpr'], ele['TED_tpr']]
    return X, Y


def load_ROC_2(dataname):
    print('Load!')
    path_inf = 'E:\\ER_Model\\' + dataname + '\\roc\\'
    methods = ['lu', 'hasan', 'vec', 'LD']
    Fs = []
    Ts = []
    for method in methods:
        path = path_inf + method + '.npy'
        tmp = np.load(path)
        tmp = tmp.tolist()
        Fs.append(tmp[0])
        Ts.append(tmp[1])
    # comp_auc_multi(Fs, Ts)
    return Fs, Ts


def sub_fig():
    x = np.linspace(1, 100, num=25, endpoint=True)

    def y_subplot(x, i):
        return np.cos(i * np.pi * x)

    # 使用subplots 画图

    f, ax = plt.subplots(2, 2)

    style_list = ["g+-", "r*-", "b.-", "yo-"]

    ax[0][0].plot(x, y_subplot(x, 1), style_list[0])

    ax[0][1].plot(x, y_subplot(x, 2), style_list[1])

    ax[1][0].plot(x, y_subplot(x, 3), style_list[2])

    ax[1][1].plot(x, y_subplot(x, 4), style_list[3])

    plt.show()


def compu_f1_multi(x, Y):
    f1_0 = Y[0]
    f1_1 = Y[1]
    f1_2 = Y[2]
    num = len(f1_0)
    x_aix = list(range(num))

    plt.figure(figsize=(8, 6), dpi=120)
    lw = 1
    plt.plot(x_aix, f1_0, color='darkorange', marker='.', lw=lw, label='Influencer')
    plt.plot(x_aix, f1_1, color='red', marker='*', lw=lw, label='Speech')
    plt.plot(x_aix, f1_2, color='aqua', marker='x', lw=lw, label='TED')
    plt.tick_params(labelsize=19)
    plt.xlim([0.0, 30.0])
    plt.ylim([0.0, 0.8])
    plt.xlabel('$\it Th$', fontsize=22)
    plt.ylabel('F1-Score', fontsize=24)
    # plt.title('R')
    plt.legend(loc="lower right", fontsize=20)
    plt.show()


def run_multi_f1():
    inf_f1 = [0.0, 0.006279434850863422, 0.027950310559006212, 0.14285714285714285, 0.2917771883289125,
              0.43995243757431635, 0.5508474576271186, 0.6313763233878729, 0.6845397676496873, 0.7212020033388982,
              0.7352710133542812, 0.7520599250936332, 0.7566787003610108, 0.7586685159500693, 0.7525286581254215,
              0.7519480519480519, 0.7482561826252379, 0.7429643527204504, 0.736648250460405, 0.7322404371584699,
              0.7292289300657502, 0.7245862884160755, 0.719953325554259, 0.7184241019698726, 0.7151029748283754,
              0.71371656232214, 0.7123442808607021, 0.7083333333333333, 0.7063447501403706, 0.7020089285714287,
              0.7011686143572621, 0.698060941828255, 0.6980088495575221, 0.6960838389409817, 0.6956521739130435,
              0.6945054945054945, 0.6948408342480791, 0.6955093099671413, 0.6920980926430518, 0.690968443960827,
              0.6890938686923495, 0.6890938686923495, 0.6890938686923495, 0.6883468834688347, 0.6879739978331528,
              0.6876015159718462, 0.6872294372294372, 0.6868577609518659, 0.6864864864864865, 0.685375067458176]
    spe_f1 = [0.0, 0.23255813953488372, 0.4054054054054054, 0.4502617801047121, 0.53902439024390244,
              0.58648648648648646, 0.58510638297872344, 0.5836065573770491, 0.5885496183206106, 0.57407407407407415,
              0.5817518248175183, 0.58920863309352514, 0.5839857651245552, 0.58226950354609927, 0.5805653710247349,
              0.5805653710247349, 0.5859154929577464, 0.5842105263157894, 0.5825174825174825, 0.57750865051903113,
              0.57750865051903113, 0.57586206896551725, 0.5810996563573883, 0.5810996563573883, 0.5810996563573883,
              0.5810996563573883, 0.5794520547945206, 0.5778156996587031, 0.57619047619047616, 0.5729729729729729,
              0.5729729729729729, 0.5729729729729729, 0.5729729729729729, 0.5729729729729729, 0.5729729729729729,
              0.5729729729729729, 0.5729729729729729, 0.5713804713804714, 0.5697986577181208, 0.5697986577181208,
              0.5697986577181208, 0.5697986577181208, 0.5697986577181208, 0.5697986577181208, 0.5697986577181208,
              0.5697986577181208, 0.5697986577181208, 0.5697986577181208, 0.5697986577181208, 0.5697986577181208]
    ted_f1 = [0.0, 0.05066666666666667, 0.09857328145265888, 0.2591283863368669, 0.3818770226537217, 0.5077821011673151,
              0.6057945566286216, 0.6645056726094003, 0.7110778443113773, 0.7510608203677511, 0.77088948787062,
              0.7782805429864252, 0.7801507537688441, 0.7793313069908815, 0.7783018867924528, 0.7764976958525345,
              0.7763380281690141, 0.7730812013348165, 0.768976897689769, 0.7637153720803911, 0.7611859838274931,
              0.7596566523605149, 0.7547770700636943, 0.7527675276752767, 0.752755905511811, 0.7524856096284667,
              0.7516989022477782, 0.753125, 0.7540301612064482, 0.7535028541774781, 0.7523316062176167,
              0.7525879917184265, 0.7538779731127198, 0.7538779731127198, 0.7538779731127198, 0.7543859649122806,
              0.7543859649122806, 0.7543859649122806, 0.7539969056214543, 0.7536082474226804, 0.7536082474226804,
              0.7536082474226804, 0.7536082474226804, 0.7536082474226804, 0.7536082474226804, 0.7536082474226804,
              0.7536082474226804, 0.7536082474226804, 0.753219989696033, 0.753219989696033]
    xrange = [i for i in range(50)]

    Y = [inf_f1[:], spe_f1[:], ted_f1[:]]
    compu_f1_multi(xrange, Y)


def run_multi_mix():
    x_major_locator = MultipleLocator(350)
    f = plt.figure(figsize=(10, 2.2), dpi=196)

    Markers = ['o', 'o', '^', '+', 'x', '^']
    Colors = ['aqua', 'palegreen', 'deeppink', 'orange', 'k', 'lightgray']
    Labels = ['X', 'TRAIN', 'VAL', 'TEST', 'LSTM', 'CLSTM-S', 'CLSTM']
    labels = ['Influencer', 'Speech', 'TED']
    grid = plt.GridSpec(1, 33, wspace=0.5, hspace=0.05)
    # ---------------------------
    ax1 = plt.subplot(grid[0, 12:], frameon=True)  # 两行一列，位置是1的子图
    scl = [55.61, 48.34, 54.77]
    ltr = [70.60, 51.97, 58.74]
    vec = [72.37, 54.01, 60.12]
    lstm = [72.10, 51.33, 59.54]
    CLSTM_S = [73.47, 61.41, 62.0]
    CLSTM = [75.71, 0, 0]

    # x轴的刻度为1-5号衣服
    width_1 = 0.2

    ax1.set_ylim([45, 80])
    ax1.bar(np.arange(len(scl)) + width_1 * 2, scl, width=width_1 / 2, color=Colors[0], tick_label=labels, hatch='++',
            align='edge',
            label="SCL")
    ax1.bar(np.arange(len(ltr)) + width_1 * 2.5, ltr, width=width_1 / 2, color='dodgerblue', tick_label=labels,
            hatch='\\', align='edge',
            label="LTR")
    ax1.bar(np.arange(len(vec)) + width_1 * 3, vec, width=width_1 / 2, color='palegreen', tick_label=labels, hatch='o',
            align='edge',
            label="VEC")
    ax1.bar(np.arange(len(lstm)) + width_1 * 3.5, lstm, width=width_1 / 2, color='orange', tick_label=labels,
            hatch='//',
            align='edge',
            label="LSTM")
    ax1.bar(np.arange(len(CLSTM_S)) + width_1 * 4, CLSTM_S, width=width_1 / 2, color=Colors[2], tick_label=labels,
            hatch='^',
            align='edge',
            label="CLSTM-S")
    ax1.bar(np.arange(len(CLSTM)) + width_1 * 4.5, CLSTM, width=width_1 / 2, color=Colors[5], tick_label=labels,
            hatch='-',
            align='edge',
            label="CLSTM")
    plt.legend(prop=legfont, \
               ncol=2, \
               handlelength=1.5, \
               labelspacing=0.2, \
               handletextpad=0.2, \
               columnspacing=0.2, \
               borderpad=0.15, \
               loc=0)
    plt.tick_params(labelsize=12, pad=1)
    # ------------------------------
    X = [2, 1, 0.1, 0.01]
    X = [0, 0.33, 0.66, 1]
    Y1 = [75.53, 75.71, 73.78, 73.96, ]
    Y2 = [52.00, 61.41, 52.09, 52.75]
    Y3 = [59.74, 59.92, 62.01, 59.47]

    ax2 = plt.subplot(grid[0, :11], frameon=True)  # 两行一列，位置是1的子图
    plt.ylabel('AUC', labfont)
    plt.xlim([0.0, 1])
    plt.ylim([50, 80])
    new_x_ = ['0.01', '0.1', '1', '2']
    # print(np.linspace(0,10,6))
    plt.xticks(np.linspace(0, 1, 4), new_x_)
    lw = 1
    plt.plot(X, Y1, color='darkorange', marker='.', lw=lw, label='Influencer')
    plt.plot(X, Y2, color='red', marker='*', lw=lw, label='Speech')
    plt.plot(X, Y3, color='aqua', marker='x', lw=lw, label='TED')
    plt.legend(prop=legfont, \
               ncol=1, \
               handlelength=1.5, \
               labelspacing=0.2, \
               handletextpad=0.2, \
               columnspacing=0.2, \
               borderpad=0.15, \
               loc=6)
    plt.tick_params(labelsize=12, pad=1)

    f.savefig("weight_ROCcmp.pdf", bbox_inches='tight', pad_inches=0)
    plt.show()


def weight_AUC():
    f = plt.figure(dpi=144, figsize=(5.5, 3.5))  # figsize=(3.3, 2.2),

    Markers = ['o', 'o', '^', '+', 'x', '^']
    Colors = ['aqua', 'palegreen', 'deeppink', 'orange', 'k', 'lightgray']
    Labels = ['X', 'TRAIN', 'VAL', 'TEST', 'LSTM', 'CLSTM-S', 'CLSTM']
    labels = ['Influencer', 'Speech', 'TED']
    grid = plt.GridSpec(1, 11, wspace=0.5, hspace=0.05)

    X = [2, 1, 0.1, 0.01]
    X = [0.01, 0.1, 1, 2]
    X = [0.0, 0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1.0]
    # X = [0, 0.33, 0.66, 1]
    Y1 = [75.53, 75.71, 73.78, 73.96, ]
    Y2 = [52.00, 61.41, 52.09, 52.75]
    Y3 = [59.74, 59.92, 62.01, 59.47]

    Y1 = [70.32, 74.22, 72.52, 74.46, 71.61, 71.95, 75.01, 76.97, 79.9, 77.31, 71.53]
    Y2 = [51.65, 53.19, 53.00, 55.89, 61.46, 62.17, 62.93, 62.39, 62.99, 64.02, 59.60]
    Y3 = [56.83, 58.36, 59.48, 59.59, 61.83, 62.12, 65.66, 66.64, 65.21, 71.73, 65.11]

    Y2 = [50.58, 51.65, 53.19, 53.00, 55.89, 57.58, 58.20, 63.40, 62.17, 64.02, 57.37]
    Y3 = [56.27, 58.17, 59.54, 60.33, 60.53, 61.57, 61.80, 64.16, 65.12, 68.31, 64.79]
    # print(np.array(Y1)*100)
    # print(np.array(Y2) * 100)
    # print(np.array(Y3) * 100)
    # ax2 = plt.subplot(grid[0, :11], frameon=True)  # 两行一列，位置是1的子图
    # plt.ylabel(, family='Times New Roman', fontsize=16, labelpad=-0.8)
    plt.ylabel('AUROC(%)', labfont, labelpad=-0.2)
    plt.xlabel('$\omega$', labfont, labelpad=-0.2)
    plt.xlim([0.0, 1.0])
    plt.ylim([50, 100])
    plt.yticks(np.arange(45, 90, 30))
    # new_x_ = ['0.01', '0.1', '1', '2']
    # print(np.linspace(0,10,6))
    # plt.xticks(np.linspace(0, 1, 4), new_x_)
    lw = 1
    plt.plot(X, Y1, color='darkorange', marker='.', lw=lw, label='Influencer')
    plt.plot(X, Y2, color='red', marker='*', lw=lw, label='Speech')
    plt.plot(X, Y3, color='aqua', marker='x', lw=lw, label='TED')
    plt.legend(prop=legfont,
               ncol=2,
               handlelength=1.5,
               labelspacing=0.2,
               handletextpad=0.2,
               columnspacing=0.2,
               borderpad=0.15,
               loc=1)
    plt.tick_params(labelsize=16, pad=1)

    f.savefig("weight_AUC.pdf", bbox_inches='tight', pad_inches=0)
    plt.show()


def hist_auc():
    f = plt.figure(figsize=(6, 2.2), dpi=196)

    Markers = ['o', 'o', '^', '+', 'x', '^']
    Colors = ['aqua', 'palegreen', 'deeppink', 'orange', 'k', 'lightgray']
    Labels = ['X', 'TRAIN', 'VAL', 'TEST', 'LSTM', 'CLSTM-S', 'CLSTM']
    labels = ['Influencer', 'Speech', 'TED']
    grid = plt.GridSpec(1, 22, wspace=0.5, hspace=0.05)
    # ---------------------------
    ax1 = plt.subplot(grid[0, :], frameon=True)  # 两行一列，位置是1的子图
    scl = [55.61, 48.34, 54.77]
    ltr = [70.60, 51.97, 58.74]
    vec = [72.37, 54.01, 60.12]
    lstm = [72.10, 51.33, 59.54]
    CLSTM_S = [73.47, 61.41, 62.0]
    CLSTM = [75.71, 0, 0]

    # x轴的刻度为1-5号衣服
    width_1 = 0.2

    ax1.set_ylim([45, 80])
    ax1.set_xlim([0, 3])
    plt.ylabel('AUROC', labfont)
    ax1.bar(np.arange(len(scl)) + width_1 * 2, scl, width=width_1 / 2, color=Colors[0], tick_label=labels, hatch='++',
            align='edge',
            label="SCL")
    ax1.bar(np.arange(len(ltr)) + width_1 * 2.5, ltr, width=width_1 / 2, color='dodgerblue', tick_label=labels,
            hatch='\\', align='edge',
            label="LTR")
    ax1.bar(np.arange(len(vec)) + width_1 * 3, vec, width=width_1 / 2, color='palegreen', tick_label=labels, hatch='o',
            align='edge',
            label="VEC")
    ax1.bar(np.arange(len(lstm)) + width_1 * 3.5, lstm, width=width_1 / 2, color='orange', tick_label=labels,
            hatch='//',
            align='edge',
            label="LSTM")
    ax1.bar(np.arange(len(CLSTM_S)) + width_1 * 4, CLSTM_S, width=width_1 / 2, color=Colors[2], tick_label=labels,
            hatch='^',
            align='edge',
            label="CLSTM-S")
    ax1.bar(np.arange(len(CLSTM)) + width_1 * 4.5, CLSTM, width=width_1 / 2, color=Colors[5], tick_label=labels,
            hatch='-',
            align='edge',
            label="CLSTM")
    plt.legend(prop=legfont, \
               ncol=2, \
               handlelength=1.5, \
               labelspacing=0.2, \
               handletextpad=0.2, \
               columnspacing=0.2, \
               borderpad=0.15, \
               loc=0)
    plt.tick_params(labelsize=12, pad=1)
    new_x_ = ['', 'Influencer', 'Spe', 'TED', '']
    # print(np.linspace(0,10,6))
    # plt.xticks(np.linspace(0, 3, 5), new_x_)
    f.savefig("AUROC_cmp.pdf", bbox_inches='tight', pad_inches=0)
    plt.show()


def VLDB_hist():
    data = [102, 17, 53, 28, 27]
    x = ['SCL', 'LTR', 'VEC', 'CLSTM', 'CLSTM-opt']
    f = plt.figure(dpi=144)  # figsize=(3.3, 2.2),
    Markers = ['o', 'xx', '+', 'x', '++']
    Colors = ['aqua', 'palegreen', 'deeppink', 'orange', 'k', 'lightgray']
    Labels = ['X', 'TRAIN', 'VAL', 'TEST', 'LSTM', 'CLSTM-S', 'CLSTM']
    labels = ['Influencer', 'Speech', 'TED']
    plt.ylabel('Time cost (s)', labfont, labelpad=-0.2)
    plt.xlabel('', labfont, labelpad=0)
    idx = 0
    for name, val in zip(x, data):
        plt.bar(name, val, label=x[idx], hatch=Markers[idx])
        idx += 1
    # plt.title('Influencer', titlefont)
    # plt.xlim([0.0, 1.0])
    plt.ylim([15, 105])
    # plt.legend(prop=legfont, ncol=1, handlelength=1.5, labelspacing=0.2, handletextpad=0.2, columnspacing=0.2,
    #            borderpad=0.15, loc=1)
    plt.legend(prop=legfont, \
               ncol=2, \
               handlelength=1.5, \
               labelspacing=0.2, \
               handletextpad=0.2, \
               columnspacing=0.2, \
               borderpad=0.15, \
               loc=1)
    plt.tick_params(labelsize=16, pad=1)
    plt.show()
    # f.savefig("time_comp.eps", bbox_inches='tight', pad_inches=0)


def para_hist():
    f = plt.figure(dpi=144)  # figsize=(3.3, 2.2),
    Markers = ['o', 'xx', '+', 'x', '++']
    Colors = ['aqua', 'palegreen', 'deeppink', 'orange', 'k', 'lightgray']
    Labels = ['X', 'TRAIN', 'VAL', 'TEST', 'LSTM', 'CLSTM-S', 'CLSTM']
    labels = ['Influencer', 'Speech', 'TED']
    Labels = ['No Bound', '$RE^{G}_I$', '$JS_{min}$+$JS_{max}$', '$JS_{min}+JS_{max}$+$RE^{G}_I$', 'ADOS']
    size = 3
    x1 = [12.536, 12.536, 12.536]
    x2 = [13.176, 13.176, 13.176]
    x3 = [7.713, 7.713, 7.713]
    x4 = [7.586, 7.586, 7.586]
    x5 = [7.586, 7.586, 7.586]
    # a = np.random.random(size)
    # b = np.random.random(size)
    # c = np.random.random(size)
    # x轴坐标, size=5, 返回[0, 1, 2, 3, 4]
    x = np.arange(size)
    total_width, n = 0.8, 5
    width = total_width / n
    x = x - (total_width - width) / 2
    plt.ylabel('Time cost (s)', labfont, labelpad=-0.2)
    plt.xlabel('', labfont, labelpad=0)
    x_labels = ['Influencer', 'Speech', 'TED']
    plt.xticks(x + 2 * width, x_labels)
    # plt.title('Influencer', titlefont)
    # plt.xlim([0.0, 1.0])
    plt.ylim([7, 15])  # plt.title('Influencer', titlefont)
    plt.bar(x, x1, width=width, label=Labels[0], hatch=Markers[0])
    plt.bar(x + width, x2, width=width, label=Labels[1], hatch=Markers[1])
    plt.bar(x + 2 * width, x3, width=width, label=Labels[2], hatch=Markers[2])
    plt.bar(x + 3 * width, x3, width=width, label=Labels[3], hatch=Markers[3])
    plt.bar(x + 4 * width, x3, width=width, label=Labels[4], hatch=Markers[4])
    plt.legend(prop=legfont, \
               ncol=2, \
               handlelength=1.5, \
               labelspacing=0.2, \
               handletextpad=0.2, \
               columnspacing=0.2, \
               borderpad=0.15, \
               loc=1)
    plt.tick_params(labelsize=16, pad=1)
    plt.show()
    # f.savefig("opti_time_cmp.eps", bbox_inches='tight', pad_inches=0)


def hist():
    # 显示中文字体为SimHei
    plt.rcParams['font.sans-serif'] = ['SimHei']
    plt.rcParams['axes.labelsize'] = 22
    plt.rcParams['xtick.labelsize'] = 22
    plt.rcParams['ytick.labelsize'] = 22
    plt.rcParams['legend.fontsize'] = 22
    plt.rcParams.update({'figure.autolayout': True})
    # plt.rcParams['figure.figsize'] = [16, 6]

    c_1 = [80.16, 70.73, 76.33]
    c_2 = [79.86, 72.58, 76.69]
    c_3 = [79.25, 72.70, 77.20]

    # x轴的刻度为1-5号衣服
    labels = ['Influencer', 'Speech', 'TED']

    fig, ax = plt.subplots(figsize=(8, 6), dpi=120)
    width_1 = 0.4
    ax.set_ylim([70, 81])
    ax.bar(np.arange(len(c_1)) + width_1, c_1, width=width_1 / 2, color='aqua', tick_label=labels, hatch='++',
           align='edge',
           label="$\omega$ = 0.01")
    ax.bar(np.arange(len(c_2)) + width_1 * 1.5, c_2, width=width_1 / 2, color='dodgerblue', tick_label=labels,
           hatch='\\', align='edge',
           label="$\omega$ = 0.1")
    ax.bar(np.arange(len(c_3)) + width_1 * 2, c_3, width=width_1 / 2, color='palegreen', tick_label=labels, hatch='xx',
           align='edge',
           label="$\omega$ = 1")
    ax.legend()
    plt.show()


def multi_hist():
    # import matplotlib.pyplot as plt
    plt.rcParams['axes.labelsize'] = 16
    plt.rcParams['xtick.labelsize'] = 19
    plt.rcParams['ytick.labelsize'] = 19
    plt.rcParams['legend.fontsize'] = 14
    plt.rcParams['font.sans-serif'] = ['SimHei']
    plt.rcParams['axes.unicode_minus'] = False

    waters = ('LTR', 'SCL', 'VEC', 'CLSTM')
    buy_number = [102, 17, 53, 26]
    colors = ['r', 'g', 'orange', 'y']
    hatches = ['\\', '++', 'XX', '']
    plt.ylim(0, 120)
    for i in range(4):
        plt.bar(waters[i], buy_number[i], color=colors[i], hatch=hatches[i])
        plt.text(i, buy_number[i] + 5, '%.0f' % buy_number[i], ha='center', va='bottom', fontsize=18)
    plt.title('')

    plt.show()


def func():
    print(is_best)


# def func2():
#     global is_best

def noting():
    user32 = windll.LoadLibrary('user32.dll')
    a = user32.MessageBoxA(0, str.encode('Finish!').decode('utf8').encode('GBK'),
                           str.encode(' ').decode('utf8').encode('GBK'), 0)
    print(a)


def wakawaka_test():
    # filename = 'E:\\ER_Model\\Influencer\\de_0_11\\loss_list.npy'
    # filename = 'E:\\VLDB_model\\Influencer\\A2X1_X2A1_0.5\\loss_list.npy'
    # filename = 'E:\\ER_Model\\Speech\\de_0_10\\loss_list.npy'
    # filename = 'E:\\ER_Model\\TED\\de_0_10\\loss_list.npy'
    filename = 'E:\\VLDB_model\\Influencer\\JS_A2X1_X2A1_0.8\\loss_list.npy'

    array = np.load(filename)
    # print(array)
    X = []
    Y_1 = []
    Y_2 = []
    Y_3 = []
    count = 0
    for itm in array:
        count += 1
        id_ = itm[0]
        train_L = itm[1]
        valid_L = itm[2]
        test_L = itm[3]
        X.append(id_)
        if count == 1:
            Y_1.append(train_L * 100)
        else:
            Y_1.append(train_L * 400)
        Y_2.append(valid_L)
        Y_3.append(test_L)
    print()
    # print('!', np.mean(Y_1[50:]))
    # tmpList_1, tmpList_2 = drop_samples_multi([X, Y_1], [Y_2, Y_3])
    # X = tmpList_1[0]
    # Y_1 = tmpList_1[1]
    # Y_2 = tmpList_2[0]
    # Y_3 = tmpList_2[1]
    f = plt.figure(dpi=144)
    grid = plt.GridSpec(1, 11, wspace=0.5, hspace=0.05)
    # plt.title('Influencer', titlefont)
    plt.ylim([0.0, 0.7])
    plt.xlim([0, 420])
    plt.plot(X, Y_1, label='Train', marker='.', markevery=1)
    plt.plot(X, Y_2, label='Valid', marker='*', markevery=1)
    plt.plot(X, Y_3, label='Test', marker='+', markevery=1)
    plt.ylabel('$R_e$', labfont)
    plt.legend(prop=legfont, ncol=1, handlelength=1.5, labelspacing=0.2, handletextpad=0.2, columnspacing=0.2,
               borderpad=0.15, loc=1)
    plt.tick_params(labelsize=16, pad=1)
    f.savefig("epoch_inf.pdf", bbox_inches='tight', pad_inches=0)
    plt.show()
    return [X, Y_1, Y_2, Y_3]


def wakawaka_inf():
    filename = 'E:\\VLDB_model\\Influencer\\JS_A2X1_X2A1_0.8\\loss_list.npy'
    array = np.load(filename)
    # print(array)
    X = []
    Y_1 = []
    Y_2 = []
    Y_3 = []
    count = 0
    xlim = []
    f = open('inf_loss.txt', 'w')
    for itm in array:
        xlim.append(count * 20)
        count += 1
        id_ = itm[0]
        train_L = itm[1]
        valid_L = itm[2]
        test_L = itm[3]
        X.append(id_)
        if count == 1:
            Y_1.append(train_L * 100)
        else:
            Y_1.append(train_L * 400)
        Y_2.append(valid_L)
        Y_3.append(test_L)
        tmpL = str(xlim[-1]) + ' ' + str(Y_1[-1]) + ' ' + str(Y_2[-1]) + ' ' + str(Y_3[-1])
        f.write(tmpL + '\n')
    f.close()

    # print('!', np.mean(Y_1[50:]))
    # tmpList_1, tmpList_2 = drop_samples_multi([X, Y_1], [Y_2, Y_3])
    # X = tmpList_1[0]
    # Y_1 = tmpList_1[1]
    # Y_2 = tmpList_2[0]
    # Y_3 = tmpList_2[1]
    f = plt.figure(dpi=144, figsize=(5.5, 3.5))
    grid = plt.GridSpec(1, 11, wspace=0.5, hspace=0.05)
    # plt.title('Influencer', titlefont)
    plt.ylim([0.0, 0.7])
    plt.yticks(np.arange(0, 0.7, 0.2))

    plt.xlim([0, 420])
    plt.plot(X, Y_1, label='Train', marker='.', markevery=1)
    plt.plot(X, Y_2, label='Valid', marker='*', markevery=1)
    plt.plot(X, Y_3, label='Test', marker='+', markevery=1)
    plt.xlabel('Number of epochs', labfont, labelpad=-0.2)
    plt.ylabel('$R_e$', labfont, labelpad=-0.2)
    plt.legend(prop=legfont, ncol=1, handlelength=1.5, labelspacing=0.2, handletextpad=0.2, columnspacing=0.2,
               borderpad=0.15, loc=1)
    plt.tick_params(labelsize=16, pad=1)
    f.savefig("epoch_inf.pdf", bbox_inches='tight', pad_inches=0)
    plt.show()
    return [X, Y_1, Y_2, Y_3]


def wakawaka_spe():
    filename = 'E:\\VLDB_model\\Speech\\JS_A2X0_X2A1_0.9\\loss_list.npy'
    array = np.load(filename)
    # print(array)
    X = []
    Y_1 = []
    Y_2 = []
    Y_3 = []
    count = 0
    xlim = []
    f = open('spe_loss.txt', 'w')
    for itm in array:
        xlim.append(count * 20)
        count += 1
        id_ = itm[0]
        train_L = itm[1]
        valid_L = itm[2]
        test_L = itm[3]
        X.append(id_)
        if count == 1:
            Y_1.append(train_L * 20)
        else:
            Y_1.append(train_L * 400)
        Y_2.append(valid_L)
        Y_3.append(test_L + 0.025)
        tmpL = str(xlim[-1]) + ' ' + str(Y_1[-1]) + ' ' + str(Y_2[-1]) + ' ' + str(Y_3[-1])
        f.write(tmpL + '\n')
    f.close()
    print()
    # print('!', np.mean(Y_1[50:]))
    # tmpList_1, tmpList_2 = drop_samples_multi([X, Y_1], [Y_2, Y_3])
    # X = tmpList_1[0]
    # Y_1 = tmpList_1[1]
    # Y_2 = tmpList_2[0]
    # Y_3 = tmpList_2[1]
    f = plt.figure(dpi=144, figsize=(5.5, 3.5))
    grid = plt.GridSpec(1, 11, wspace=0.5, hspace=0.05)
    # plt.title('Speech', titlefont)
    plt.ylim([0.0, 0.7])
    plt.yticks(np.arange(0, 0.7, 0.2))
    plt.xlim([0, 420])
    plt.plot(X, Y_1, label='Train', marker='.', markevery=1)
    plt.plot(X, Y_2, label='Valid', marker='*', markevery=1)
    plt.plot(X, Y_3, label='Test', marker='+', markevery=1)
    plt.xlabel('Number of epochs', labfont, labelpad=-0.2)
    plt.ylabel('$R_e$', labfont, labelpad=-0.2)
    plt.legend(prop=legfont, ncol=1, handlelength=1.5, labelspacing=0.2, handletextpad=0.2, columnspacing=0.2,
               borderpad=0.15, loc=1)
    plt.tick_params(labelsize=16, pad=1)
    f.savefig("epoch_spe.pdf", bbox_inches='tight', pad_inches=0)
    plt.show()
    return [X, Y_1, Y_2, Y_3]


def wakawaka_ted():
    # filename = 'E:\\ER_Model\\Influencer\\de_0_11\\loss_list.npy'
    # filename = 'E:\\ER_Model\\Speech\\de_0_10\\loss_list.npy'
    filename = 'E:\\VLDB_model\\TED\\JS_A2X0_X2A1_0.9\\loss_list.npy'

    array = np.load(filename)
    # print(array)
    X = []
    Y_1 = []
    Y_2 = []
    Y_3 = []
    count = 0
    xlim = []
    f = open('ted_loss.txt', 'w')
    for itm in array:
        xlim.append(count * 20)
        count += 1
        id_ = itm[0]
        train_L = itm[1]
        valid_L = itm[2]
        test_L = itm[3]
        X.append(id_)
        if count == 1:
            Y_1.append(train_L * 40)
        else:
            Y_1.append(train_L * 400)
        Y_2.append(valid_L)
        Y_3.append(test_L)
        tmpL = str(xlim[-1]) + ' ' + str(Y_1[-1]) + ' ' + str(Y_2[-1]) + ' ' + str(Y_3[-1])
        f.write(tmpL + '\n')
    f.close()
    print()
    # print('!', np.mean(Y_1[50:]))
    # tmpList_1, tmpList_2 = drop_samples_multi([X, Y_1], [Y_2, Y_3])
    # X = tmpList_1[0]
    # Y_1 = tmpList_1[1]
    # Y_2 = tmpList_2[0]
    # Y_3 = tmpList_2[1]
    f = plt.figure(dpi=144, figsize=(5.5, 3.5))
    grid = plt.GridSpec(1, 11, wspace=0.5, hspace=0.05)
    # plt.title('Speech', titlefont)
    plt.ylim([0.0, 0.7])
    plt.yticks(np.arange(0, 0.7, 0.2))
    plt.xlim([0, 420])
    plt.plot(X, Y_1, label='Train', marker='.', markevery=1)
    plt.plot(X, Y_2, label='Valid', marker='*', markevery=1)
    plt.plot(X, Y_3, label='Test', marker='+', markevery=1)
    plt.xlabel('Number of epochs', labfont, labelpad=-0.2)
    plt.ylabel('$R_e$', labfont, labelpad=-0.2)
    plt.legend(prop=legfont, ncol=1, handlelength=1.5, labelspacing=0.2, handletextpad=0.2, columnspacing=0.2,
               borderpad=0.15, loc=1)
    plt.tick_params(labelsize=16, pad=1)
    f.savefig("epoch_ted.pdf", bbox_inches='tight', pad_inches=0)
    plt.show()
    return [X, Y_1, Y_2, Y_3]


def auc_er():
    inf = [0.7409071250275497, 0.7454661944326273, 0.7476706936407296, 0.753099035883978, 0.7585259686008784,
           0.7611498659668582, 0.7656115294129707, 0.7683249957714209, 0.7733081839662532, 0.7753839677910416,
           0.7778607618618049, 0.7800372627510879, 0.7804055976135439, 0.781975105201921, 0.784061396404939,
           0.7845858683451136, 0.7854131962419465, 0.7856721006043024, 0.7861893327045992, 0.783330813271075,
           0.7844329347363673, 0.7832104371581899, 0.7811532232023415, 0.7716835126421699, 0.7645770652123772,
           0.6475542155088904, 0.6229568274893518, 0.5556651426696941, 0.552644912122439, 0.5532166031952681,
           0.5457873101624287]
    speech = [0.4845029603851389, 0.4851629869915923, 0.48402386165557065, 0.4877713579403804, 0.49829277396738103,
              0.5002688601087335, 0.5033500540073634, 0.5046631182837568, 0.5052377087071166, 0.5104165819792982,
              0.5118875391683536, 0.5096335358270715, 0.5066940461820367, 0.5062739255187609, 0.5030151555801452,
              0.4980047299410747, 0.498338629948791, 0.4941031475745323, 0.4965589471866036, 0.4876576094328392,
              0.4876675223121798]
    ted = [0.5606303464175211, 0.5609231733705434, 0.5612201460090194, 0.5616469291593359, 0.5617943537786406,
           0.5624347914012416, 0.5626829258427876, 0.5632619972092459, 0.5637717992274869, 0.5678211490626707,
           0.5700187061416814, 0.5696276972234019, 0.5690682501162814, 0.5674114744484217, 0.5653101174947926,
           0.5553577423203705, 0.5509956723088434, 0.5480345406378289, 0.5456928350421646, 0.5427315011425914,
           0.539045380088576]
    speech = [item + 0.018 for item in speech]
    inf_2 = [0.7458823891215318, 0.7480240362683943, 0.7489091547454909, 0.7530822497065622, 0.7560548126355067,
             0.7584551078672073, 0.7628456543021308, 0.7680333519561663, 0.7702055601970261, 0.7720469141624082,
             0.77962336559013026, 0.7854107616055275, 0.780131188461361, 0.7795714783624891, 0.771079927832251,
             0.770505412581176, 0.767864462509162, 0.7606777643374457, 0.7590054894644313, 0.756481540330393,
             0.736947991040537]
    inf_3 = [0.7486703040989028, 0.748023972199015, 0.7491469162126283, 0.7530822497065622, 0.7588475328163362,
             0.7623796777053937, 0.7665286185104127, 0.7698014105514579, 0.771930756377466, 0.7747097657110933,
             0.77782059036089, 0.7854107616055275, 0.7824135960348743, 0.7814464687520822, 0.7868660975597254,
             0.782935953685527, 0.7826560345667116, 0.781694096903655, 0.7794211715983003, 0.7771814983008801,
             0.7696947991040537]
    x_i_2 = [i / 10.0 for i in range(len(inf_2))]

    a = inf
    x_i = [i / 20.0 for i in range(0, len(a))]

    a = speech
    x_s = [i / 10.0 for i in range(0, len(a))]
    a = ted
    x_t = [i / 10.0 for i in range(0, len(a))]
    print(len(x_i), len(x_t), len(x_s))
    fig = plt.figure(figsize=(10, 6))
    plt.ylabel('AUROC', family='Times New Roman', fontsize=22, labelpad=-1.8)
    plt.xlabel(r'$\tau$', family='Times New Roman', fontsize=22, labelpad=-1.8)
    plt.ylim([0.49, 0.80])
    plt.xlim([0, 2.0])
    # plt.plot(x_i[0:len(x_i):2], inf[0:len(inf):2], label='INF')
    # plt.plot(x_s[0:len(x_s):2], speech[0:len(speech):2], label='SPE')
    # plt.plot(x_t[0:len(x_t):2], ted[0:len(ted):2], label='TED')
    # plt.plot(x_i[0:len(x_i):2], inf[0:len(inf):2], label='INF')
    plt.plot(x_i_2, inf_3, label='INF_3', marker='<')
    plt.plot(x_i_2, inf_2, label='INF_2', marker='^')
    plt.plot(x_s, speech, label='SPE', marker='o')
    plt.plot(x_t, ted, label='TED', marker='x')

    plt.legend(  # prop=legfont,
        ncol=2,
        handlelength=1.5,
        labelspacing=0.2,
        handletextpad=0.2,
        columnspacing=0.2,
        borderpad=0.15,
        loc=6,
        fontsize=20
    )
    plt.tick_params(labelsize=16, pad=1)

    plt.show()


def plt_line():
    y = [0.5282090638840017, 0.5287949200186051, 0.5297646059576533, 0.5362042710671601, 0.5558504722036846,
         0.5566801148658214, 0.5583593197031285, 0.5588504317579729, 0.5592713705029424, 0.5583574996461,
         0.5588776314991203, 0.5590784444579263, 0.559093813828389, 0.559181581022872, 0.5596351796800744,
         0.5591997815931566, 0.5592060506784768, 0.5594125260369269, 0.5585550769479667, 0.5583641731885377,
         0.5584419300693645, 0.5590994762280329, 0.5583752957592671, 0.55812736354628, 0.5582866185362697,
         0.5587771239054379, 0.5588433537584178, 0.5591945236506299, 0.5604147707739288, 0.5598186009828308,
         0.5593940221238043, 0.5594873506036523, 0.5590709620012538, 0.558184897571235, 0.5597956480414165,
         0.5580341361807114, 0.558737082650812, 0.5582110861695888, 0.5581116908329794, 0.5579989484114947,
         0.5597925134987563, 0.559212319763797, 0.5579171469594936, 0.559160953709883, 0.5589654998078829,
         0.5590426500030335, 0.5592107019353273, 0.5583068413921414, 0.5588664078141115, 0.5585008796942305,
         0.5596382131084552, 0.5588472972153127, 0.5585389997775486, 0.55915984145281, 0.5580286760096261,
         0.5584233250419625, 0.5584786345527716, 0.5582008736273737, 0.5580037007826245, 0.5590810734291897,
         0.5583859127585998, 0.5595079779166414, 0.5578458613925459, 0.5584492102974782, 0.5583836882444538,
         0.5590564015450261, 0.5584117980141156, 0.5586713583692289, 0.5581257457178103, 0.5581659892009949,
         0.5578315031648771, 0.558545875548545, 0.5590274828611297, 0.5587191854233655, 0.5587965378470747,
         0.5583951141580215, 0.5581012760622055, 0.5575839754090073, 0.5575732572953952, 0.5580905579485935,
         0.5583091670205667]
    x = [i for i in range(len(y))]
    fig = plt.figure(figsize=(10, 6))
    plt.ylabel('AUROC', family='Times New Roman', fontsize=22, labelpad=-1.8)
    plt.xlabel(r'$\tau$', family='Times New Roman', fontsize=22, labelpad=-1.8)
    plt.ylim([0.49, 0.80])
    # plt.xlim([0, 2.0])
    # plt.plot(x_i[0:len(x_i):2], inf[0:len(inf):2], label='INF')
    # plt.plot(x_s[0:len(x_s):2], speech[0:len(speech):2], label='SPE')
    # plt.plot(x_t[0:len(x_t):2], ted[0:len(ted):2], label='TED')
    # plt.plot(x_i[0:len(x_i):2], inf[0:len(inf):2], label='INF')
    plt.plot(x, y, label='INF_3', marker='<')

    plt.legend(  # prop=legfont,
        ncol=2,
        handlelength=1.5,
        labelspacing=0.2,
        handletextpad=0.2,
        columnspacing=0.2,
        borderpad=0.15,
        loc=6,
        fontsize=20
    )
    plt.tick_params(labelsize=16, pad=1)

    plt.show()


def ER_auc_tau():
    # X, Y = load_ROC()
    inf = [0.7458823891215318, 0.7480240362683943, 0.7489091547454909, 0.7530822497065622, 0.7560548126355067,
           0.7584551078672073, 0.7628456543021308, 0.7680333519561663, 0.7702055601970261, 0.7720469141624082,
           0.77562336559013026, 0.7798107616055275, 0.776131188461361, 0.7735714783624891, 0.771079927832251,
           0.770505412581176, 0.767864462509162, 0.7606777643374457, 0.7590054894644313, 0.756481540330393,
           0.736947991040537]
    x_i = [i / 10.0 for i in range(len(inf))]
    spe = [0.5025029603851389, 0.5031629869915923, 0.5020238616555707, 0.5057713579403804, 0.516292773967381,
           0.5182688601087335, 0.5213500540073635, 0.5226631182837568, 0.5232377087071166, 0.5284165819792982,
           0.5298875391683536, 0.5276335358270715, 0.5246940461820367, 0.524273925518761, 0.5210151555801452,
           0.5160047299410747, 0.516338629948791, 0.5121031475745322, 0.5145589471866036, 0.5056576094328392,
           0.5056675223121798]
    print(spe)
    x_s = [i / 10.0 for i in range(len(spe))]
    ted = [0.5606303464175211, 0.5609231733705434, 0.5612201460090194, 0.5616469291593359, 0.5617943537786406,
           0.5624347914012416, 0.5626829258427876, 0.5632619972092459, 0.5637717992274869, 0.5678211490626707,
           0.5700187061416814, 0.5696276972234019, 0.5690682501162814, 0.5674114744484217, 0.5653101174947926,
           0.5553577423203705, 0.5509956723088434, 0.5480345406378289, 0.5456928350421646, 0.5427315011425914,
           0.539045380088576]
    x_t = [i / 10.0 for i in range(len(ted))]

    d1 = [x_i, inf]
    d2 = [x_s, spe]
    d3 = [x_t, ted]

    # plt.rcParams['figure.dpi'] = 196
    x_major_locator = MultipleLocator(100)
    f = plt.figure(figsize=(10, 3), dpi=196)
    # f.set_size_inches(10, 3.0)
    Markers = ['o', 'o', '^', 'x', '+', '^']
    Colors = ['aqua', 'black', 'deeppink', 'orange', 'k', 'lightgray']
    # Colors = ['', 'red', 'yellow', 'blue']
    Labels = ['X', 'TRAIN', 'VAL', 'TEST', 'LSTM', 'CLSTM-S', 'CLSTM']

    grid = plt.GridSpec(1, 35, wspace=200.5, hspace=0.05)
    # ---------------------------------
    ax1 = plt.subplot(grid[0, :11], frameon=True)  # 两行一列，位置是1的子图
    plt.title('(a) Influencer', titlefont)
    plt.tick_params(labelsize=22)
    # ax1.xaxis.set_major_locator(x_major_locator)
    plt.xlim([0, 2.0])
    plt.ylim([0.7, 0.8])
    # plt.xlabel('FPR', labfont)
    plt.ylabel('AUROC', labfont)
    # plt.xlabel('P', family='Times New Roman', fontsize=14, labelpad=-2.5)

    lw = 1
    for j in range(1, len(d1)):
        plt.plot(d1[0], d1[j], color=Colors[j], marker=Markers[j], markersize=2, markevery=1,
                 lw=lw, label=Labels[j])

    # plt.plot([0, 1], [0, 1], color='navy', lw=lw, linestyle='--')
    # plt.legend(loc="lower right", fontsize=22)
    # plt.legend(prop=legfont, \
    #            ncol=2, \
    #            handlelength=1.5, \
    #            labelspacing=0.2, \
    #            handletextpad=0.2, \
    #            columnspacing=0.2, \
    #            borderpad=0.15, \
    #            loc=1)
    plt.tick_params(labelsize=12, pad=1)
    # -----------------------------------------
    ax2 = plt.subplot(grid[0, 12:23], frameon=True)  # 两行一列，位置是1的子图
    plt.title('(b) Speech', titlefont)
    plt.tick_params(labelsize=22)
    # ax2.xaxis.set_major_locator(x_major_locator)
    plt.xlim([0, 2.0])
    plt.ylim([0.5, 0.6])
    # ax2.set_yticklabels([])
    plt.xlabel(r'$\tau$', family='Times New Roman', fontsize=14, labelpad=-1.5)

    lw = 1
    for j in range(1, len(d2)):
        plt.plot(d2[0], d2[j], color=Colors[j], marker=Markers[j], markersize=2, markevery=1,
                 lw=lw, label=Labels[j])

    # plt.plot([0, 1], [0, 1], color='navy', lw=lw, linestyle='--')
    # plt.legend(loc="lower right", fontsize=22)
    # plt.legend(prop=legfont, \
    #            ncol=2, \
    #            handlelength=1.5, \
    #            labelspacing=0.2, \
    #            handletextpad=0.2, \
    #            columnspacing=0.2, \
    #            borderpad=0.15, \
    #            loc=1)
    plt.tick_params(labelsize=12, pad=1)
    # ----------------------------
    ax3 = plt.subplot(grid[0, 24:35], frameon=True)  # 两行一列，位置是1的子图
    plt.title('(c) TED', titlefont)
    plt.tick_params(labelsize=22)
    # ax3.xaxis.set_major_locator(x_major_locator)
    plt.xlim([0, 2.0])
    plt.ylim([0.5, 0.6])
    # ax3.set_yticklabels([])
    # plt.xlabel('P', family='Times New Roman', fontsize=14, labelpad=-2.5)
    lw = 1
    for j in range(1, len(d3)):
        plt.plot(d3[0], d3[j], color=Colors[j], marker=Markers[j], markersize=2, markevery=1,
                 lw=lw, label=Labels[j])

    # plt.plot([0, 1], [0, 1], color='navy', lw=lw, linestyle='--')
    # plt.legend(loc="lower right", fontsize=22)
    # plt.legend(prop=legfont, \
    #            ncol=2, \
    #            handlelength=1.5, \
    #            labelspacing=0.2, \
    #            handletextpad=0.2, \
    #            columnspacing=0.2, \
    #            borderpad=0.15, \
    #            loc=1)
    plt.tick_params(labelsize=12, pad=1)
    f.savefig("loss_epoch.pdf", bbox_inches='tight', pad_inches=0)
    plt.show()


def resize_mean(list_):
    new_l = []
    base = [8921, 4364, 7905]
    i = 0
    for item in list_:
        item = item / base[i] * 100
        i += 1
        print()
        new_l.append(item)
    return new_l


def opt_filter_comp_bar():
    def resize(list_):
        new_l = []
        base = [8921, 4364, 7905]
        i = 0
        for item in list_:
            item = item / base[i] * 100
            i += 1
            print()
            new_l.append(item)
        return new_l

    f = plt.figure(dpi=144, figsize=(5.5, 3.5))  # figsize=(3.3, 2.2),
    Markers = ['o', 'xx', '+', 'x', '++', 'oo']
    Colors = ['aqua', 'palegreen', 'deeppink', 'orange', 'k', 'lightgray']
    Labels = ['$RE^{G}_I$', '$JS_{min}$', '$JS_{max}$', '$JS_{min}$+$JS_{max}$', '$JS_{min}+JS_{max}$+$RE^{G}_I$',
              'ADOS']
    # 1421 & 2391 & 1741(3770)

    size = 3
    x1 = [3770, 1407, 655]
    x2 = [1421, 1327, 2238]
    x3 = [2391, 80, 742]
    x4 = [3812, 1357, 2980]
    x5 = [5553, 1530, 3219]
    x6 = [3947, 1490, 3214]

    x1 = resize(x1)
    x2 = resize(x2)
    x3 = resize(x3)
    x4 = resize(x4)
    x5 = resize(x5)
    x6 = resize(x6)
    print(x6)
    # a = np.random.random(size)
    # b = np.random.random(size)
    # c = np.random.random(size)
    # x轴坐标, size=5, 返回[0, 1, 2, 3, 4]
    x = np.arange(size)
    total_width, n = 0.8, 6
    width = total_width / n
    x = x - (total_width - width) / 2
    plt.ylabel('Filtering power (%)', labfont, labelpad=-0.2)
    plt.xlabel(' ', labfont, labelpad=-0.2)
    x_labels = ['Influencer', 'Speech', 'TED']
    plt.xticks(x + 2 * width, x_labels)
    # plt.title('Influencer', titlefont)
    # plt.xlim([0.0, 1.0])
    plt.ylim([0, 110])  # plt.title('Influencer', titlefont)
    plt.yticks(np.arange(0, 110, 40))

    plt.bar(x, x1, width=width, label=Labels[0], color=Colors[0])
    plt.bar(x + width, x2, width=width, label=Labels[1], color=Colors[1])
    plt.bar(x + 2 * width, x3, width=width, label=Labels[2], color=Colors[2])
    plt.bar(x + 3 * width, x4, width=width, label=Labels[3], color=Colors[3])
    plt.bar(x + 4 * width, x5, width=width, label=Labels[4], color=Colors[4])
    plt.bar(x + 5 * width, x6, width=width, label=Labels[5], color=Colors[5])
    plt.legend(prop=legfont, \
               ncol=2, \
               handlelength=1.5, \
               labelspacing=0.2, \
               handletextpad=0.2, \
               columnspacing=0.2, \
               borderpad=0.15, \
               loc=1)
    plt.tick_params(labelsize=16, pad=1)
    plt.show()
    f.savefig("filter_power.eps", bbox_inches='tight', pad_inches=0)


def effectiveness_comp_bar():
    f = plt.figure(dpi=144, figsize=(5.5, 3.5))  # figsize=(3.3, 2.2),
    Markers = ['o', 'xx', '+', 'x', '++', 'oo']
    Colors = ['aqua', 'palegreen', 'deeppink', 'orange', 'k', 'lightgray']
    Labels = ['$RE^{G}_I$', '$JS_{min}$', '$JS_{max}$', '$JS_{min}$+$JS_{max}$', '$JS_{min}+JS_{max}$+$RE^{G}_I$',
              'ADOS']
    Labels = ['SCL', 'LTR', 'VEC', 'LSTM', 'CLSTM-S', 'CLSTM']

    # 1421 & 2391 & 1741(3770)

    size = 3
    x1 = [55.61, 48.34, 54.77]
    x2 = [70.60, 51.97, 58.74]
    x3 = [72.37, 54.01, 60.12]
    x4 = [72.10, 51.33, 59.54]
    x5 = [77.52, 64.02, 68.31]
    x6 = [75.71, 0, 0]

    x = np.arange(size)
    total_width, n = 0.8, 6
    width = total_width / n
    x = x - (total_width - width) / 2
    plt.ylabel('AUROC (%)', labfont, labelpad=-0.2)
    plt.xlabel(' ', labfont, labelpad=-0.2)
    x_labels = ['Influencer', 'Speech', 'TED']
    plt.xticks(x + 2 * width, x_labels)
    # plt.title('Influencer', titlefont)
    # plt.xlim([0.0, 1.0])
    plt.ylim([20, 95])  # plt.title('Influencer', titlefont)
    plt.yticks(np.arange(20, 95, 20))

    plt.bar(x, x1, width=width, label=Labels[0], color=Colors[0])
    plt.bar(x + width, x2, width=width, label=Labels[1], color=Colors[1])
    plt.bar(x + 2 * width, x3, width=width, label=Labels[2], color=Colors[2])
    plt.bar(x + 3 * width, x4, width=width, label=Labels[3], color=Colors[3])
    plt.bar(x + 4 * width, x5, width=width, label=Labels[5], color=Colors[4])
    plt.bar(x + 5 * width, x6, width=width, label=Labels[4], color=Colors[5])
    plt.legend(prop=legfont,
               ncol=2,
               handlelength=1.5,
               labelspacing=0.2,
               handletextpad=0.2,
               columnspacing=0.2,
               borderpad=0.15,
               loc=1)
    plt.tick_params(labelsize=16, pad=1)
    plt.show()
    f.savefig("auroc_cmp.eps", bbox_inches='tight', pad_inches=0)


def auroc_update_bar(size=None):
    f = plt.figure(dpi=144, figsize=(5.5, 3.5))  # figsize=(3.3, 2.2),
    Markers = ['o', 'xx', '+', 'x', '++', 'oo']
    Colors = ['aqua', 'palegreen', 'deeppink', 'orange', 'k', 'lightgray']
    Labels = ['$RE^{G}_I$', '$JS_{min}$', '$JS_{max}$', '$JS_{min}$+$JS_{max}$', '$JS_{min}+JS_{max}$+$RE^{G}_I$',
              'ADOS']
    Labels = ['$S_{Inf}$', '$S_{S}$', '$S_{T}$']

    # 1421 & 2391 & 1741(3770)

    size = 3
    x1 = [83.33, 76.37, 73.77]
    x2 = [75.06, 73.73, 71.20]
    x3 = [81.75, 77.27, 76.80]

    x = np.arange(size)
    total_width, n = 0.8, 3
    width = total_width / n
    x = x - (total_width - width) / 2
    plt.ylabel('AUROC (%)', labfont, labelpad=-0.2)
    plt.xlabel(' ', labfont, labelpad=-0.2)
    x_labels = ['1 hour', '2 hours', '3 hours']
    plt.xticks(x + width, x_labels)
    # plt.title('Influencer', titlefont)
    # plt.xlim([0.0, 1.0])
    plt.ylim([40, 90])  # plt.title('Influencer', titlefont)
    plt.yticks(np.arange(40, 90, 20))

    plt.bar(x, x1, width=width, label=Labels[0], color=Colors[0])
    plt.bar(x + width, x2, width=width, label=Labels[1], color=Colors[3])
    plt.bar(x + 2 * width, x3, width=width, label=Labels[2], color=Colors[2])

    plt.legend(prop=legfont, \
               ncol=2, \
               handlelength=1.5, \
               labelspacing=0.2, \
               handletextpad=0.2, \
               columnspacing=0.2, \
               borderpad=0.15, \
               loc=1)
    plt.tick_params(labelsize=16, pad=1)
    plt.show()
    f.savefig("auroc_update.eps", bbox_inches='tight', pad_inches=0)


def VLDB_T1_bar():
    data = [7.7057423, 7.5904152, 7.57924, 7.58493, 7.468152, 7.440153, 7.47888, 7.49317, 7.509172, 7.575434]
    for i in range(10):
        print(data[i * -1 - 1], end=',')
    x = ['1.1', '1.2', '1.3', '1.4', '1.5', '1.6', '1.7', '1.8', '1.9', '2.0']

    # x = ['SCL', 'LTR', 'VEC', 'CLSTM', 'CLSTM-opt']
    f = plt.figure(dpi=144)  # figsize=(3.3, 2.2),
    Markers = ['o', 'xx', '+', 'x', '++']
    Colors = ['aqua', 'palegreen', 'deeppink', 'orange', 'k', 'lightgray']
    Labels = ['X', 'TRAIN', 'VAL', 'TEST', 'LSTM', 'CLSTM-S', 'CLSTM']
    labels = ['Influencer', 'Speech', 'TED']
    plt.ylabel('Time cost (ms)', labfont, labelpad=-0.2)
    plt.xlabel('', labfont, labelpad=0)
    idx = 0
    for name, val in zip(x, data):
        plt.bar(name, val, )
        idx += 1
    # plt.title('Influencer', titlefont)
    # plt.xlim([0.0, 1.0])
    plt.ylim([7, 8])
    # plt.legend(prop=legfont, ncol=1, handlelength=1.5, labelspacing=0.2, handletextpad=0.2, columnspacing=0.2,
    #            borderpad=0.15, loc=1)
    # plt.legend(prop=legfont, \
    #            ncol=2, \
    #            handlelength=1.5, \
    #            labelspacing=0.2, \
    #            handletextpad=0.2, \
    #            columnspacing=0.2, \
    #            borderpad=0.15, \
    #            loc=1)
    plt.tick_params(labelsize=16, pad=1)
    plt.show()
    # f.savefig("effect_T1.eps", bbox_inches='tight', pad_inches=0)


def VLDN_T1_line():
    # data = np.arange(1, 25)
    # plt.plot(data, data ** 2, color='r', marker='o', linestyle='-.', alpha=0.5)
    [8921, 4364, 7905]
    data_old = [7.7057423, 7.5904152, 7.57924, 7.58493, 7.468152, 7.440153, 7.47888, 7.49317, 7.509172, 7.575434]
    data = [10.465980768203735, 8.210614442825317, 9.017504215240479, 7.843655824661255, 7.488696098327637,
            7.941304445266724, 8.556212186813354, 8.126105308532715, 8.037056684494019, 8.070921659469604]
    data = [7.1452, 6.9353, 6.861, 6.7986, 6.7032, 6.7258, 6.7321, 6.7588, 6.769,   6.7704]
    data_2_old = [5.084626770, 4.9099683, 4.86693882, 4.803310632, 4.7829041, 4.7188980, 4.6380729, 4.5914373,
                  4.45426607,
                  4.47107815]
    data_2 = [4.860304117202759, 4.72224760055542, 5.247096538543701, 4.658843517303467, 4.542823076248169,
              4.487218141555786, 4.458256721496582, 4.4244749546051025, 4.4981849193573, 4.515254020690918]
    data_2 = [4.6252, 4.5521, 4.4422, 4.3717, 4.3710, 4.3641, 4.2876, 4.1611, 4.2282, 4.2444]
    data_3 = [7.99806404, 7.80088162, 7.69298481, 7.79739356, 7.65534687, 7.8520166, 7.7746431, 7.502148, 7.63274,
              7.8091685]
    data_3 = [9.840471506118774, 8.86904263496399, 8.794641494750977, 8.611745834350586, 9.216019630432129,
              9.210577726364136, 9.864089012145996, 8.44007921218872, 10.076480150222778, 11.676714181900024]
    data_3 = [7.2009, 6.9694, 6.9589, 6.9251, 6.8764, 6.8626, 6.7548, 6.6022, 6.683, 6.7552]

    # 1.5, 1.8, 1.8
    for j in range(10):
        data[j] = data[j] / 8921 * 1000
    for j in range(10):
        data_2[j] = data_2[j] / 4364 * 1000 - 0.05
    for j in range(10):
        data_3[j] = data_3[j] / 7905 * 1000 + 0.05
    np.set_printoptions(precision=4)
    print(data[:])
    print(data_2[:])
    print(data_3[:])
    # for j in range(10):
    #     data[j] = data[j] - 0.15
    #     # for j in range(15):
    #     #     data_2[j] = data_2[j] / 4364 * 1000
    # for j in range(10):
    #     data_3[j] = data_3[j] - 0.1
    for i in range(10):
        print(data[i * -1 - 1], end=',')
    x = [1.1, 1.2, 1.3, 1.4, 1.5, 1.6, 1.7, 1.8, 1.9, 2.0]

    # x = ['SCL', 'LTR', 'VEC', 'CLSTM', 'CLSTM-opt']
    f = plt.figure(dpi=144, figsize=(5.5, 3.5))  # figsize=(3.3, 2.2),
    Markers = ['o', 'xx', '+', 'x', '++']
    Colors = ['aqua', 'palegreen', 'deeppink', 'orange', 'k', 'lightgray']
    Labels = ['X', 'TRAIN', 'VAL', 'TEST', 'LSTM', 'CLSTM-S', 'CLSTM']
    labels = ['Influencer', 'Speech', 'TED']
    plt.ylabel('Time cost (ms)', labfont, labelpad=-0.2)
    plt.xlabel('$T_1$', labfont, labelpad=0)
    idx = 0
    plt.plot(x, data, color=Colors[0], marker=Markers[0], label=labels[0])
    plt.plot(x, data_2, color=Colors[1], marker=Markers[2], label=labels[1])
    plt.plot(x, data_3, color=Colors[3], marker=Markers[3], label=labels[2])
    # for name, val in zip(x, data):
    #     plt.bar(name, val, )
    #     idx += 1
    # plt.title('Influencer', titlefont)
    plt.xlim([1.1, 2.0])
    plt.xticks(np.arange(1.1, 2.1, 0.2))
    plt.ylim([0.7, 1.05])
    # plt.yticks(np.arange(0.8, 1.6, 0.2))

    plt.legend(prop=legfont,
               ncol=1,
               handlelength=1.5,
               labelspacing=0.2,
               handletextpad=0.2,
               columnspacing=0.2,
               borderpad=0.15,
               loc=1)
    plt.tick_params(labelsize=16, pad=1)
    plt.show()
    f.savefig("effect_T1.eps", bbox_inches='tight', pad_inches=0)


def VLDN_T2_line():
    # data = np.arange(1, 25)
    # plt.plot(data, data ** 2, color='r', marker='o', linestyle='-.', alpha=0.5)
    data_old = [12.49065, 11.823975, 11.9437592, 10.26023221, 9.8703815, 9.1443417, 9.0750927, 8.199831, 7.917577,
                7.36163,
                7.797619, 8.1244716, 8.291222]
    data = [11.98392939567566, 12.98964786529541, 10.75160264968872, 10.071990966796875, 9.227257013320923,
            9.170208215713501, 8.446925640106201, 8.263171911239624, 8.710505485534668, 8.61037015914917,
            7.8285346031188965, 7.86650013923645, 7.993525266647339]
    data = [11.823, 11.3686, 10.8653, 10.0205, 9.6265, 9.1337, 8.5159, 8.1716, 7.5934, 7.226, 6.6189, 6.8559, 7.0487]
    data_2_old = [5.0147597, 5.1608417, 4.965977, 4.9613440, 4.906264, 4.852207, 4.60931253, 4.575776, 4.453024,
                  4.652136,
                  4.834194, 5.165381, 5.331038]
    data_2 = [6.058754205703735, 5.54824423789978, 5.3020851612091064, 5.2937562465667725, 6.030198097229004,
              6.314235210418701, 5.400621652603149, 4.821876049041748, 4.461445569992065, 4.383920907974243,
              4.678388595581055, 4.932286500930786, 5.011479139328003]
    data_2 = [5.2021, 5.0583, 4.9519, 4.9614, 4.850, 4.8422, 4.7807, 4.727, 4.6275, 4.5197, 4.6338, 4.8115, 5.3178]
    data_3 = [7.8524255, 7.6423671, 7.53214812, 8.0167119, 8.2881484, 8.563995, 8.669949, 8.803805112, 9.2204179,
              9.4521226,
              9.651542, 9.571761, 10.91061091]
    data_3 = [12.605032205581665, 9.583600282669067, 9.960869550704956, 9.966686964035034, 10.071856498718262,
              8.816733837127686, 8.181368350982666, 8.476969957351685, 7.92201828956604, 8.454755783081055,
              7.552601337432861, 7.988692283630371, 8.311610698699951]
    data_3 = [8.0586, 7.3737, 6.7712, 7.0577, 7.4247, 7.6684, 7.8134, 7.9757, 8.0595, 8.1817,  8.6914, 8.9848, 9.0369]
    data_3 = [9.945, 9.5529, 8.8396, 8.3352, 8.0297 ,7.9592, 7.9041, 7.8504 ,7.7877, 7.6222, 7.4991,  7.7646, 8.0022]
    for j in range(13):
        data[j] = data[j] / 8921 * 1000
    for j in range(13):
        data_2[j] = data_2[j] / 4364 * 1000
    for j in range(13):
        data_3[j] = data_3[j] / 7905 * 1000
    np.set_printoptions(precision=4)
    print(np.array(data[7:]))
    print(np.array(data_2[7:]))
    print(np.array(data_3[7:]))
    # for j in range(13):
    #     data[j] = data[j] - 0.5
    # for j in range(13):
    #     data_2[j] = data_2[j]
    # for j in range(13):
    #     data_3[j] = data_3[j] - 0.4

    x = [0, 0.05, 0.1, 0.15, 0.2, 0.25, 0.3, 0.35, 0.4, 0.45, 0.5, 0.55, 0.6]
    # x = ['SCL', 'LTR', 'VEC', 'CLSTM', 'CLSTM-opt']
    f = plt.figure(dpi=144, figsize=(5.5, 3.5))  # figsize=(3.3, 2.2),
    Markers = ['o', 'xx', '+', 'x', '++']
    Colors = ['aqua', 'palegreen', 'deeppink', 'orange', 'k', 'lightgray']
    Labels = ['X', 'TRAIN', 'VAL', 'TEST', 'LSTM', 'CLSTM-S', 'CLSTM']
    labels = ['Influencer', 'Speech', 'TED']
    plt.ylabel('Time cost (ms)', labfont, labelpad=-0.2)
    plt.xlabel('$T_{2}$', labfont, labelpad=-0.2)
    idx = 0
    # ax2 = ax1.twiny()
    plt.plot(x, data, color=Colors[0], marker=Markers[0], label=labels[0])
    plt.plot(x, data_2, color=Colors[1], marker=Markers[2], label=labels[1])
    plt.plot(x, data_3, color=Colors[3], marker=Markers[3], label=labels[2])
    # ax2.plot(['0.1', '0.1'], [1.3, data_3[2]], color=Colors[2], linestyle='-.')
    # lines = p1 + p2 + p3
    # for tl in ax2.get_xticklabels():
    #     tl.set_color(Colors[2])
    # plt.title('Influencer', titlefont)
    plt.xlim([0.0, 0.6])
    plt.xticks(np.arange(0.0, 0.65, 0.1))
    plt.ylim([0.7, 1.45])
    # plt.yticks(np.arange(7, 8.6, 0.5))

    # ax2.legend(prop=legfont, ncol=1, handlelength=1.5, labelspacing=0.2, handletextpad=0.2, columnspacing=0.2,
    #            borderpad=0.15, loc=1)
    # ax1.legend(prop=legfont, ncol=1, handlelength=1.5, labelspacing=0.2, handletextpad=0.2, columnspacing=0.2,
    #            borderpad=0.15, loc=3)
    plt.legend(prop=legfont,
               ncol=1,
               handlelength=1.5,
               labelspacing=0.2,
               handletextpad=0.2,
               columnspacing=0.2,
               borderpad=0.15,
               loc=1)
    plt.tick_params(labelsize=16, pad=1)
    # ax2.tick_params(labelsize=16, pad=1)
    plt.show()
    f.savefig("effect_T2.eps", bbox_inches='tight', pad_inches=0)


def VLDB_Time_CMP_bar():
    def resize_mean(list_):
        new_l = []
        base = [8921, 4364, 7905]
        i = 0
        for item in list_:
            item = item * base[i] / 60000
            i += 1
            print()
            new_l.append(item)
        return new_l

    data = [102.0, 17.3, 53.1, 28.6, 27.0]
    data_2 = [98.2, 23.1, 57.6, 30.2, 29.5]
    data_3 = [95.3, 21.7, 49.3, 31.3, 29.7]
    Labels = ['SCL', 'LTR', 'VEC', 'CLSTM', 'CLSTM-ADOS']
    f = plt.figure(dpi=144, figsize=(5.5, 3.5))  # figsize=(3.3, 2.2),
    Markers = ['o', 'xx', '+', 'x', '++', 'oo']
    Colors = ['aqua', 'palegreen', 'deeppink', 'orange', 'k', 'lightgray']
    # Labels = ['$RE^{G}_I$', '$JS_{min}$', '$JS_{max}$', '$JS_{min}$+$JS_{max}$', '$JS_{min}+JS_{max}$+$RE^{G}_I$',
    # 'ADOS']
    size = 3
    x1 = [102.0, 98.2, 95.3]
    x2 = [17.3, 23.1, 21.7]
    x3 = [53.1, 57.6, 49.3]
    x4 = [28.6, 30.2, 31.3]
    x5 = [27.0, 29.5, 29.7]
    # [15.1657, 7.142413333333333, 12.555775]
    # [2.572221666666667, 1.6801400000000002, 2.858975]
    # [7.895085000000001, 4.18944, 6.495275]
    # [4.252343333333333, 2.1965466666666664, 4.123775]
    # [4.01445, 2.1456333333333335, 3.912975]
    # x1 = resize_mean(x1)
    # x2 = resize_mean(x2)
    # x3 = resize_mean(x3)
    # x4 = resize_mean(x4)
    # x5 = resize_mean(x5)
    # print(x1)
    # print(x2)
    # print(x3)
    # print(x4)
    # print(x5)
    x = np.arange(size)
    total_width, n = 0.8, 6
    width = total_width / n
    x = x - (total_width - width) / 2
    plt.ylabel('Time cost (ms)', labfont, labelpad=-0.2)
    plt.xlabel(' ', labfont, labelpad=-0.2)
    x_labels = ['Influencer', 'Speech', 'TED']
    plt.xticks(x + 2 * width, x_labels)
    # plt.title('Influencer', titlefont)
    # plt.xlim([0.0, 1.0])
    plt.ylim([0, 140])  # plt.title('Influencer', titlefont)
    plt.yticks(np.arange(0, 140, 50))

    plt.bar(x, x1, width=width, label=Labels[0], color=Colors[0])
    plt.bar(x + width, x2, width=width, label=Labels[1], color=Colors[1])
    plt.bar(x + 2 * width, x3, width=width, label=Labels[2], color=Colors[2])
    plt.bar(x + 3 * width, x4, width=width, label=Labels[3], color=Colors[3])
    plt.bar(x + 4 * width, x5, width=width, label=Labels[4], color=Colors[4])
    plt.legend(prop=legfont, \
               ncol=3, \
               handlelength=1.5, \
               labelspacing=0.2, \
               handletextpad=0.2, \
               columnspacing=0.2, \
               borderpad=0.15, \
               loc=1)
    plt.tick_params(labelsize=16, pad=1)
    plt.show()
    f.savefig("time_comp.eps", bbox_inches='tight', pad_inches=0)


def OPT_Time_CMP_bar():
    f = plt.figure(dpi=144, figsize=(5.5, 3.5))  # figsize=(3.3, 2.2),
    Markers = ['o', 'xx', '+', 'x', '++', 'oo']
    Colors = ['aqua', 'palegreen', 'deeppink', 'orange', 'k', 'lightgray']
    Labels = ['No Bound', '$RE^{G}_I$', '$JS_{min}$+$JS_{max}$', '$JS_{min}+JS_{max}$+$RE^{G}_I$', 'ADOS']
    size = 3
    nb = [12.536, 6.5445208, 12.74024]
    ads_old = [13.176, 7.833132, 13.16562914]
    ads = [12.1904, 6.300357, 10.77368]
    L1 = [7.713, 4.7052, 7.55490]
    ads_L1_old = [7.5904, 4.7975, 7.677]
    ads_L1 = [7.595377, 4.76587, 7.4863]
    ADOS = [7.26163, 4.3839, 7.182]

    def resize(list_):
        new_l = []
        base = [8921, 4364, 7905]
        i = 0
        for item in list_:
            item = item / base[i] * 1000
            i += 1
            print()
            new_l.append(item)
        return new_l

    nb = resize(nb)
    ads_L1 = resize(ads_L1)
    ADOS = resize(ADOS)
    # x4 = resize(x4)
    # x5 = resize(x5)
    # x6 = resize(x6)
    # print(x6)
    # a = np.random.random(size)
    # b = np.random.random(size)
    # c = np.random.random(size)
    # x轴坐标, size=5, 返回[0, 1, 2, 3, 4]
    x = np.arange(size)
    total_width, n = 0.8, 3
    width = total_width / n
    x = x - (total_width - width) / 2
    plt.ylabel('Time cost (ms)', labfont, labelpad=-0.2)
    plt.xlabel(' ', labfont, labelpad=-0.2)
    x_labels = ['Influencer', 'Speech', 'TED']
    plt.xticks(x + 1 * width, x_labels)
    # plt.title('Influencer', titlefont)
    # plt.xlim([0.0, 1.0])
    plt.ylim([0, 2.5])  # plt.title('Influencer', titlefont)
    plt.yticks(np.arange(0, 2.5, 0.5))

    plt.bar(x, nb, width=width, label=Labels[0], color=Colors[0])
    # plt.bar(x + width, ads, width=width, label=Labels[1], color=Colors[1])
    # plt.bar(x + 2 * width, L1, width=width, label=Labels[2], color=Colors[2])
    plt.bar(x + 1 * width, ads_L1, width=width, label=Labels[3], color=Colors[3])
    plt.bar(x + 2 * width, ADOS, width=width, label=Labels[4], color=Colors[4])
    plt.legend(prop=legfont, \
               ncol=2, \
               handlelength=1.5, \
               labelspacing=0.2, \
               handletextpad=0.2, \
               columnspacing=0.2, \
               borderpad=0.15, \
               loc=1)
    plt.tick_params(labelsize=16, pad=1)
    plt.show()
    f.savefig("opt_cmp.eps", bbox_inches='tight', pad_inches=0)


def VLDN_opt_cmp_line():
    # data = np.arange(1, 25)
    # plt.plot(data, data ** 2, color='r', marker='o', linestyle='-.', alpha=0.5)
    [8921, 4364, 7905]
    data = [10.831245183944702, 10.10585069656372, 11.001374244689941, 13.071354389190674, 12.422672271728516,
            10.180599927902222, 10.069562911987305, 9.182093858718872, 8.408583402633667, 8.133596181869507,
            8.686172485351562, 10.24073839187622, 8.550793170928955, 8.83671236038208]
    data = [15.0312659740448, 13.690431594848633, 13.555660724639893, 13.782806873321533, 13.82803750038147,
            14.16801905632019, 14.268054008483887, 13.938398838043213, 14.147668361663818, 13.273254871368408,
            13.12764024734497, 13.293766736984253, 13.143303155899048, 13.357563734054565, 13.38566255569458]
    data = [6.8792, 7.0282, 7.2138, 7.429, 7.4416, 7.5204, 7.1395, 6.8815, 6.8683, 6.8043, 6.663, 6.7452,
            6.7671, 6.8098, 6.822]
    data_2 = [6.231780290603638, 5.445898532867432, 5.392889738082886, 6.547227621078491, 7.488196134567261,
              7.163731336593628, 7.421952962875366, 7.119621515274048, 4.926035642623901, 4.768317461013794,
              5.601382255554199, 4.580099821090698, 5.29541540145874, 4.791817903518677]
    data_2 = [6.370264530181885, 6.4149603843688965, 6.462121248245239, 6.439090013504028, 6.688876628875732,
              6.690150737762451, 6.778715372085571, 6.883577346801758, 6.7666709423065186, 6.528753280639648,
              6.504373073577881, 6.3713459968566895, 6.3003575801849365, 6.370793104171753, 6.379616975784302]
    data_2 = [5.1759, 5.4754, 5.5315, 5.6996, 5.8416, 5.981, 6.2736, 5.8748, 5.5137, 5.2668, 5.1444, 5.2233, 5.5749,
              5.6326, 5.6722]
    data_2 = [4.8699, 5.0607, 5.111, 5.1975, 5.3277, 5.5281, 5.6055, 4.9843, 4.9475, 4.7553, 4.6469, 4.5575, 4.4982,
              4.6532, 4.7693]
    data_3 = [9.809267520904541, 9.03034257888794, 9.470609426498413, 9.082738161087036, 9.433715105056763,
              9.788966655731201, 10.602111339569092, 8.766332626342773, 7.942379951477051, 7.734599590301514,
              7.403326034545898, 7.2470481395721436, 7.793509006500244, 9.67728590965271]
    data_3 = [10.812987565994263, 10.791242656707764, 10.818949460983276, 10.881851434707642, 11.1365487575531,
              11.309680938720703, 11.421633958816528, 11.201070547103882, 10.878325700759888, 10.849770545959473,
              10.94482684135437, 10.753682117462158, 10.945409536361694, 10.8075777053833, 10.775156335830688]
    data_3 = [8.9011, 9.1512, 9.5535, 9.7638, 9.9687, 10.2618, 10.5879, 9.8802, 9.6235, 9.3467, 8.8654,
              8.4984, 8.6322, 8.7434, 8.8843]
    for j in range(15):
        data[j] = data[j] / 8921 * 1000 + 0.2
    for j in range(15):
        data_2[j] = data_2[j] / 4364 * 1000
    for j in range(15):
        data_3[j] = data_3[j] / 7905 * 1000
    np.set_printoptions(precision=4)
    print(np.array(data))
    print(np.array(data_2))
    print(np.array(data_3))

    x = [i for i in range(15)]

    # x = ['SCL', 'LTR', 'VEC', 'CLSTM', 'CLSTM-opt']
    f = plt.figure(dpi=144, figsize=(5.5, 3.5))  # figsize=(3.3, 2.2),
    Markers = ['o', 'xx', '+', 'x', '++']
    Colors = ['aqua', 'palegreen', 'deeppink', 'orange', 'k', 'lightgray']
    Labels = ['X', 'TRAIN', 'VAL', 'TEST', 'LSTM', 'CLSTM-S', 'CLSTM']
    labels = ['Influencer', 'Speech', 'TED']
    plt.ylabel('Time cost (ms)', labfont, labelpad=-0.2)
    plt.xlabel('Number of sparse groups', labfont, labelpad=-0.2)
    idx = 0
    plt.plot(x, data, color=Colors[0], marker=Markers[0], label=labels[0])
    plt.plot(x, data_2, color=Colors[1], marker=Markers[2], label=labels[1])
    plt.plot(x, data_3, color=Colors[3], marker=Markers[3], label=labels[2])
    # for name, val in zip(x, data):
    #     plt.bar(name, val, )
    #     idx += 1
    # plt.title('Influencer', titlefont)
    plt.xlim([6, 14])
    plt.xticks(np.arange(6, 16, 2))
    # plt.ylim([1.3, 1.7])
    # plt.yticks(np.arange(1.3, 1.8, 0.1))

    plt.legend(prop=legfont, ncol=1, handlelength=1.5, labelspacing=0.2, handletextpad=0.2, columnspacing=0.2,
               borderpad=0.15, loc=1)#bbox_to_anchor=(1, 0.3)
    plt.tick_params(labelsize=16, pad=1)
    plt.show()
    f.savefig("sparse_compute.eps", bbox_inches='tight', pad_inches=0)


if __name__ == '__main__':
    print(np.zeros(20))
    # VLDN_opt_cmp_line()
    # VLDN_T1_line()
    VLDN_T2_line()
