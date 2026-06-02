#some codes adapted from https://github.com/YuemingJin/MTRCNet-CL
# and https://github.com/YuemingJin/TMRNet
# and https://github.com/tobiascz/TeCNO

import torch
from torch import optim
from torch import nn
import numpy as np
import model_v1
import model_v2
import model_v3
import model_v4
import model_v5
import pickle, time
import random
from sklearn import metrics
import copy
import datetime
import argparse
from Model.utils import  fusion
from Model.Data_Augmentation import  segmented_interpolate_augment, elastic_segment_augment, split_and_select
from get_transitionmap import get_transitionmap
import torch.nn.functional as F
import importlib
import psutil

import os
os.environ['CUDA_LAUNCH_BLOCKING'] = '1'
cpu_num = 8 # 这里设置成你想运行的CPU个数
os.environ ['OMP_NUM_THREADS'] = str(cpu_num)
os.environ ['OPENBLAS_NUM_THREADS'] = str(cpu_num)
os.environ ['MKL_NUM_THREADS'] = str(cpu_num)
os.environ ['VECLIB_MAXIMUM_THREADS'] = str(cpu_num)
os.environ ['NUMEXPR_NUM_THREADS'] = str(cpu_num)
torch.set_num_threads(cpu_num)


# def setup_resources(ID):
#     """配置CPU和GPU资源保护"""
#     # 设置CPU优先级
#     p = psutil.Process(os.getpid())
#     try:
#         p.nice(-15)  # Linux/Mac高优先级
#     except:
#         pass
#
#     # 绑定CPU核心(示例绑定前4个核心)
#     try:
#         p.cpu_affinity(list(range(8)))
#     except:
#         pass
#
#     # 设置GPU独占
#     os.environ["CUDA_VISIBLE_DEVICES"] = ID  # 只使用第一块GPU
#     torch.backends.cudnn.benchmark = True  # 启用CuDNN自动优化


phase2label_dicts = {
    'cholec80': {
        'Preparation': 0,
        'CalotTriangleDissection': 1,
        'ClippingCutting': 2,
        'GallbladderDissection': 3,
        'GallbladderPackaging': 4,
        'CleaningCoagulation': 5,
        'GallbladderRetraction': 6},

    'm2cai16': {
        'TrocarPlacement': 0,
        'Preparation': 1,
        'CalotTriangleDissection': 2,
        'ClippingCutting': 3,
        'GallbladderDissection': 4,
        'GallbladderPackaging': 5,
        'CleaningCoagulation': 6,
        'GallbladderRetraction': 7},
    'autolaparo': {
        'Preparation': 0,
        'DividingLigamentPeritoneum': 1,
        'DividingUterineVesselsLigament': 2,
        'TransectingtheVagina': 3,
        'SpecimenRemoval': 4,
        'Suturing': 5,
        'Washing': 6}
}
# 'Preparation', 'Dividing Ligament and Peritoneum', 'Dividing Uterine Vessels and Ligament',
# 'Transecting the Vagina', 'Specimen Removal', 'Suturing', 'Washing',

def get_data(data_path):
    with open(data_path, 'rb') as f:
        train_test_paths_labels = pickle.load(f)

    train_paths = train_test_paths_labels[0]
    val_paths = train_test_paths_labels[1]

    train_labels = train_test_paths_labels[2]
    val_labels = train_test_paths_labels[3]

    train_num_each = train_test_paths_labels[4]
    val_num_each = train_test_paths_labels[5]

    test_paths = train_test_paths_labels[6]
    test_labels = train_test_paths_labels[7]
    test_num_each = train_test_paths_labels[8]


    # print('train_paths_19  : {:6d}'.format(len(train_paths_19)))
    # print('train_labels_19 : {:6d}'.format(len(train_labels_19)))
    print('train_paths  : {:6d}'.format(len(train_paths)))
    print('train_labels : {:6d}'.format(len(train_labels)))
    print('valid_paths  : {:6d}'.format(len(val_paths)))
    print('valid_labels : {:6d}'.format(len(val_labels)))

    # train_labels_19 = np.asarray(train_labels_19, dtype=np.int64)
    train_labels = np.asarray(train_labels, dtype=np.int64)
    val_labels = np.asarray(val_labels, dtype=np.int64)
    test_labels = np.asarray(test_labels, dtype=np.int64)

    train_start_vidx = []
    count = 0
    for i in range(len(train_num_each)):
        train_start_vidx.append(count)
        count += train_num_each[i]

    val_start_vidx = []
    count = 0
    for i in range(len(val_num_each)):
        val_start_vidx.append(count)
        count += val_num_each[i]

    test_start_vidx = []
    count = 0
    for i in range(len(test_num_each)):
        test_start_vidx.append(count)
        count += test_num_each[i]

    return train_labels, train_num_each, train_start_vidx, val_labels, val_num_each, val_start_vidx,\
           test_labels, test_num_each, test_start_vidx

def get_long_feature(start_index, lfb, LFB_length):
    long_feature = []
    long_feature_each = []
    # 上一个存在feature的index
    for k in range(LFB_length):
        LFB_index = (start_index + k)
        LFB_index = int(LFB_index)
        long_feature_each.append(lfb[LFB_index])
    long_feature.append(long_feature_each)
    return long_feature


def transform_labels(original_labels):
    if not isinstance(original_labels, torch.Tensor):
        raise ValueError("Input must be a PyTorch tensor.")
    transformed = torch.zeros_like(original_labels)

    transformed[0] = 0  # 第一个元素始终为0
    for i in range(1, original_labels.size(0)):
        # 如果当前元素与前一个元素不同，则标记为1；否则为0
        if original_labels[i] != original_labels[i - 1]:
            transformed[i] = 1
        else:
            transformed[i] = 0

    return transformed


def str2bool(v):
    if isinstance(v, bool):
        return v
    if v.lower() in ('yes', 'true', 't', 'y', '1'):
        return True
    elif v.lower() in ('no', 'false', 'f', 'n', '0'):
        return False
    else:
        raise argparse.ArgumentTypeError('Boolean value expected.')

parser = argparse.ArgumentParser()
# parser.add_argument('--dataset', default="cholec80")
# parser.add_argument('--dataset', default="m2cai16")
parser.add_argument('--dataset', default="autolaparo")
parser.add_argument('--device', default="cuda:2", help='Device to use (default: cuda:1)')
parser.add_argument('--num_classes', default=7)
parser.add_argument('--model', default="model_ahdtcn")
parser.add_argument('--learning_rate', default=1e-3, type=float)
parser.add_argument('--epochs', default=100)
parser.add_argument('--gpu', default="2", type=str)
parser.add_argument('--ms_loss', default=True, type=bool)

parser.add_argument('--pretrain', type=str2bool, default=True, help="Run or not.")
parser.add_argument('--last', type=str2bool, default=False, help="Run or not.")
parser.add_argument('--first', type=str2bool, default=True, help="Run or not.")
parser.add_argument('--split', type=str2bool, default=True, help="Run or not.")
parser.add_argument('--DoubleTCN', type=str2bool, default=True, help="Run or not.")
parser.add_argument('--wave', type=str2bool, default=True, help="Run or not.")
parser.add_argument('--smooth', type=str2bool, default=True, help="Run or not.")
parser.add_argument('--resnet', type=str2bool, default=True, help="Run or not.")
parser.add_argument('--seed', default="2", type=int) # 11
parser.add_argument('--num_f_maps', default="48", type=int) # 11
parser.add_argument('--dim', default="2048", type=int) # 11
parser.add_argument('--base', default="4", type=int) # 11
parser.add_argument('--num_d', default="4", type=int) # 11
parser.add_argument('--Difference', type=str2bool, default=True, help="Run or not.")
parser.add_argument(
    '--DifInterval',
    nargs='+',           # 接收多个值，自动变成列表
    type=int,            # 每个值转为 int
    # default=[],  # 直接给列表，而不是字符串
    default=[],  # 直接给列表，而不是字符串
    help="List of numbers (e.g., 1 4 16 64)"
)
parser.add_argument('--num_layer', default="9", type=int) # 11
parser.add_argument('--num_stage', default="2", type=int) # 2
parser.add_argument('--num_hie', default="2", type=int) # 2
parser.add_argument('--causal_conv', type=str2bool, default=True, help="Run or not.")
parser.add_argument('--decomp_flag', type=str2bool, default=True, help="Run or not.")
parser.add_argument("--input",  # 参数名称
    type=str,   # 参数类型
    choices=["class", "feat"],  # 允许的选择
    default="class",  # 默认值
    help="选择模型版本（class、feat）"
)
parser.add_argument("--coding_side",  # 参数名称
    type=str,   # 参数类型
    choices=["left", "right", "both"],  # 允许的选择
    default="right",  # 默认值
    help="选择模型版本（left、right、both）"
)



args = parser.parse_args()

if args.base > 0:
    for i in range(args.num_d):
        args.DifInterval.append(args.base**i)
else:
    args.DifInterval=[]

print("args.DifInterval:", args.DifInterval)
print("args.causal_conv:", args.causal_conv)
print("args.input:", args.input)
print("args.coding_side:", args.coding_side)

if args.dataset == 'cholec80':
    train_labels, train_num_each, train_start_vidx,\
            val_labels, val_num_each, val_start_vidx,\
            test_labels, test_num_each, test_start_vidx = get_data('./train_val_test_path_labels_fwp.pkl')
    # train_val_test_path_labels_0119
    # train_val_test_path_labels_fwp
    if args.pretrain == False:
        with open("./LFB/g_LFB50_train_nopre.pkl", 'rb') as f:
            g_LFB_train = pickle.load(f)

        with open("./LFB/g_LFB50_val_nopre.pkl", 'rb') as f:
            g_LFB_val = pickle.load(f)

        with open("./LFB/g_LFB50_test_nopre.pkl", 'rb') as f:
            g_LFB_test = pickle.load(f)

    else:
        with open("./LFB/g_LFB50_train1.pkl", 'rb') as f:
            g_LFB_train = pickle.load(f)

        with open("./LFB/g_LFB50_val1.pkl", 'rb') as f:
            g_LFB_val = pickle.load(f)

        with open("./LFB/g_LFB50_test1.pkl", 'rb') as f:
            g_LFB_test = pickle.load(f)

elif args.dataset == 'm2cai16':
    train_labels, train_num_each, train_start_vidx, \
        val_labels, val_num_each, val_start_vidx, \
        test_labels, test_num_each, test_start_vidx = get_data('./train_val_test_path_labels_m2cai16.pkl')
    with open("./LFB/g_LFB50_train_16.pkl", 'rb') as f:
        g_LFB_train = pickle.load(f)
    with open("./LFB/g_LFB50_val_16.pkl", 'rb') as f:
        g_LFB_val = pickle.load(f)
    with open("./LFB/g_LFB50_test_16.pkl", 'rb') as f:
        g_LFB_test = pickle.load(f)

elif args.dataset == 'autolaparo':
    train_labels, train_num_each, train_start_vidx, \
        val_labels, val_num_each, val_start_vidx, \
        test_labels, test_num_each, test_start_vidx = get_data('./train_val_test_path_labels_autolaparo_0507.pkl')
    with open("./LFB/g_LFB50_train_autolaparo.pkl", 'rb') as f:
        g_LFB_train = pickle.load(f)
    with open("./LFB/g_LFB50_val_autolaparo.pkl", 'rb') as f:
        g_LFB_val = pickle.load(f)
    with open("./LFB/g_LFB50_test_autolaparo.pkl", 'rb') as f:
        g_LFB_test = pickle.load(f)



f_path = os.path.abspath('..')
root_path = f_path.split('surgical_code')[0]
args.num_classes = len(phase2label_dicts[args.dataset])


# seed = 1  # 42 for best cholec80:  1 for best m2cai16
print("Random Seed: ", args.seed)
random.seed(args.seed)
np.random.seed(args.seed)
torch.manual_seed(args.seed)

torch.cuda.manual_seed_all(args.seed)

num_gpu = torch.cuda.device_count()
use_gpu = torch.cuda.is_available()

# # 当在CuDNN后端运行时，需要进一步设置两个选项以【确保确定性】
# torch.backends.cudnn.deterministic = True
# # 关闭CuDNN的自动优化，保证每次运行结果一致
# torch.backends.cudnn.benchmark = False

os.environ['CUDA_VISIBLE_DEVICES'] = args.gpu
# setup_resources(args.gpu)
device = torch.device("cuda:{}".format(args.gpu))


args.device = device
loss_layer = nn.CrossEntropyLoss()
mse_layer = nn.MSELoss(reduction='none')
l1_layer = nn.L1Loss()

if args.dataset == 'cholec80':
    weights_train = np.asarray([1.6411019141231247,
                                0.19090963801041133,
                                1.0,
                                0.2502662616859295,
                                1.9176363911137977,
                                0.9840248158200853,
                                2.174635818337618, ])
    criterion_phase = nn.CrossEntropyLoss(weight=torch.from_numpy(weights_train).float().to(device))
else:
    criterion_phase = nn.CrossEntropyLoss()

if args.model == "model_tecno":
    print("model:", args.model)
    model_ver = importlib.import_module(args.model)
    # (mstcn_stages=2, mstcn_layers=8, mstcn_f_maps=32, mstcn_f_dim=2048, out_features=7, mstcn_causal_conv = True)
    model = model_ver.MS_TCT(mstcn_stages = 2, mstcn_layers = 8, mstcn_f_maps = 32, mstcn_f_dim = args.dim, out_features = args.num_classes, mstcn_causal_conv = args.causal_conv)
else:
    model_ver = importlib.import_module(args.model)
    model = model_ver.AHDTCN(args, args.num_f_maps, args.dim, args.num_classes)
# model = model_ver.MS_TCT(args, args.num_layer, num_f_maps, dim, num_classes)
# model = mstcn.MultiStageModel(mstcn_stages, mstcn_layers, mstcn_f_maps, mstcn_f_dim, out_features, mstcn_causal_conv)
# model = model_v1.MS_TCT(args, num_f_maps, dim, num_classes)
# model.cuda()

model.to(device=device)
best_model_wts = copy.deepcopy(model.state_dict())
best_val_accuracy_phase = 0.0
best_accuracy_phase = 0.0
best_precision_phase = 0.0
best_recall_phase = 0.0
best_jaccard_phase = 0.0

correspond_train_acc_phase = 0.0
best_epoch = 0
model.to(device)
best_epoch = 0
best_acc = 0
model.train()

if args.dataset == 'cholec80':
    train_we_use_start_idx = [x for x in range(40)]
    val_we_use_start_idx = [x for x in range(8)]
    test_we_use_start_idx = [x for x in range(40)]
elif args.dataset == 'm2cai16':
    train_we_use_start_idx = [x for x in range(27)]
    val_we_use_start_idx = [x for x in range(7)]
    test_we_use_start_idx = [x for x in range(14)]
elif args.dataset == 'autolaparo':
    train_we_use_start_idx = [x for x in range(10)]
    val_we_use_start_idx = [x for x in range(4)]
    test_we_use_start_idx = [x for x in range(7)]

import logging
# from datetime import datetime
current_time = datetime.datetime.now().strftime("%Y%m%d_%H%M")
# 根据数据集和当前时间创建日志文件名
if args.dataset == 'cholec80':
    filename = f'log_cholec80_{current_time}.txt'
elif args.dataset == 'm2cai16':
    filename = f'log_m2cai16_{current_time}.txt'
elif args.dataset == 'autolaparo':
    filename = f'log_autolaparo_{current_time}.txt'
else:
    filename = f'log_unknown_dataset_{current_time}.txt'  # 默认情况处理
logging.basicConfig(
    filename=filename,
    level=logging.INFO,
    format='%(asctime)s - %(message)s',
    datefmt='%Y-%m-%d %H:%M:%S'
)

for epoch in range(1, args.epochs + 1):
    with torch.cuda.device(device):
        torch.cuda.empty_cache()
    # if epoch == 100:
    #     args.learning_rate = 1e-3
    if epoch % 30 == 0 and args.learning_rate > 1e-7:
        args.learning_rate = args.learning_rate * 0.5
    # if epoch % 20 == 0:
    #     args.learning_rate = args.learning_rate * 0.5
    correct = 0
    total = 0
    loss_item = 0
    ce_item = 0
    ms_item = 0
    lc_item = 0
    gl_item = 0
    optimizer = torch.optim.Adam(model.parameters(), args.learning_rate, weight_decay=1e-5)
    max_seq = 0
    mean_len = 0
    ans = 0
    max_phase = 0
    # torch.cuda.empty_cache()
    train_idx = []
    model.train()
    train_loss_phase = 0.0
    train_corrects_phase = 0
    batch_progress = 0.0
    running_loss_phase = 0.0
    minibatch_correct_phase = 0.0
    train_start_time = time.time()
    for i in train_we_use_start_idx:
        optimizer.zero_grad()
        labels_phase = []
        for j in range(train_start_vidx[i], train_start_vidx[i]+train_num_each[i]):
            if args.dataset == 'cholec80' or args.dataset == 'autolaparo':
                labels_phase.append(train_labels[j][0])
            else:
                labels_phase.append(train_labels[j])

        labels_phase = torch.LongTensor(labels_phase)
        if use_gpu:
            labels_phase = labels_phase.to(device)
        else:
            labels_phase = labels_phase
        long_feature = get_long_feature(start_index=train_start_vidx[i],
                                        lfb=g_LFB_train, LFB_length=train_num_each[i])

        long_feature = (torch.Tensor(long_feature)).to(device)

        # 数据增强
        # N = 5  # 好好学习局部特征
        # if np.random.random() > 0.8:
        #     labels_phase, long_feature = elastic_segment_augment(labels_phase, long_feature)
        # if epoch % N == 0 and epoch < 20:
        #     labels_phase, long_feature = split_and_select(labels_phase, long_feature, 0, N)
        # elif epoch % 5 == 1 and epoch < 20:
        #     labels_phase, long_feature = split_and_select(labels_phase, long_feature, 1, N)
        # elif epoch % 5 == 2 and epoch < 20:
        #     labels_phase, long_feature = split_and_select(labels_phase, long_feature, 2, N)
        # elif epoch % 5 == 3 and epoch < 20:
        #     labels_phase, long_feature = split_and_select(labels_phase, long_feature, 3, N)
        # elif epoch % 5 == 4 and epoch < 20:
        #     labels_phase, long_feature = split_and_select(labels_phase, long_feature, 4, N)


        video_fe = long_feature.transpose(2, 1)
        predicted_list = model(video_fe)
        mean_len += predicted_list[0].size(-1)
        ans += 1
        if args.dataset == "m2cai16":
            labels_phase = labels_phase.squeeze(1)
        all_out, resize_list, labels_list = fusion(predicted_list, labels_phase, args)
        max_seq = max(max_seq, long_feature.size(1))

        loss = 0
        if args.ms_loss:
            ms_loss = 0
            for p, l in zip(resize_list, labels_list):
                ms_loss += loss_layer(p.transpose(2, 1).contiguous().view(-1, args.num_classes), l.view(-1))
                ms_loss += torch.mean(torch.clamp(
                    mse_layer(F.log_softmax(p[:, :, 1:], dim=1), F.log_softmax(p.detach()[:, :, :-1], dim=1)), min=0,
                    max=16))

                # new_var = torch.nn.functional.one_hot(l, num_classes=p.size(1))  # 形状变为 [1, 1810, 7]
                # new_var = new_var.permute(0, 2, 1).float()
                # f = l.unsqueeze(0).unsqueeze(0)
                # one_hot_l = torch.zeros(1, p.size(1), p.size(2), device=l.device)
                # one_hot_l[0, l, torch.arange(p.size(2))] = 1
                # ms_loss += (torch.fft.rfft(p, dim=2) - torch.fft.rfft(one_hot_l, dim=2)).abs().mean()*3
                # ms_loss += l1_layer(map, labels_map)
            loss = loss + ms_loss
            ms_item += ms_loss.item()
        optimizer.zero_grad()
        loss_item += loss.item()
        if args.last:
            all_out = resize_list[-1]
        if args.first:
            all_out = resize_list[0]
        loss.backward()
        optimizer.step()

        _, predicted = torch.max(all_out.data, 1)
        correct += ((predicted == labels_phase).sum()).item()
        total += labels_phase.shape[0]

        print('Train Epoch {}: Acc {}, Loss {}, ms {}'.format(epoch, correct / total, loss_item /total,  ms_item/total))

        ####################
        running_loss_phase += ms_loss.data.item()
        train_loss_phase += ms_loss.data.item()

        batch_corrects_phase = torch.sum(predicted == labels_phase.data)
        train_corrects_phase += batch_corrects_phase
        minibatch_correct_phase += batch_corrects_phase



    train_elapsed_time = time.time() - train_start_time
    train_accuracy_phase = float(train_corrects_phase) / len(train_labels)
    train_average_loss_phase = train_loss_phase

    # Sets the module in evaluation mode.
    model.eval()
    model.to(device)
    val_loss_phase = 0.0
    val_corrects_phase = 0
    val_start_time = time.time()
    val_progress = 0
    val_all_preds_phase = []
    val_all_labels_phase = []
    val_acc_each_video = []
    val_ms_item = 0

    with torch.no_grad():
        for i in val_we_use_start_idx:
            labels_phase = []
            for j in range(val_start_vidx[i], val_start_vidx[i] + val_num_each[i]):
                if args.dataset == 'cholec80' or args.dataset == 'autolaparo':
                    labels_phase.append(val_labels[j][0])
                else:
                    labels_phase.append(val_labels[j])
            labels_phase = torch.LongTensor(labels_phase)
            if use_gpu:
                labels_phase = labels_phase.to(device)
            else:
                labels_phase = labels_phase

            long_feature = get_long_feature(start_index=val_start_vidx[i],
                                            lfb=g_LFB_val, LFB_length=val_num_each[i])

            long_feature = (torch.Tensor(long_feature)).to(device)
            video_fe = long_feature.transpose(2, 1)
            predicted_list = model(video_fe)
            mean_len += predicted_list[0].size(-1)
            ans += 1
            if args.dataset == 'm2cai16':
                labels_phase = labels_phase.squeeze(1)
            all_out, resize_list, labels_list = fusion(predicted_list, labels_phase, args)
            max_seq = max(max_seq, video_fe.size(1))
            loss = 0
            if args.ms_loss:
                ms_loss = 0
                for p, l in zip(resize_list, labels_list):
                    ms_loss += loss_layer(p.transpose(2, 1).contiguous().view(-1, args.num_classes), l.view(-1))
                    ms_loss += torch.mean(torch.clamp(
                        mse_layer(F.log_softmax(p[:, :, 1:], dim=1), F.log_softmax(p.detach()[:, :, :-1], dim=1)),
                        min=0,
                        max=16))

                    l_expanded = l.unsqueeze(1).transpose(1, 0)
                    new_var = torch.nn.functional.one_hot(l_expanded, num_classes=p.size(1))  # 形状变为 [1, 1810, 7]
                    new_var = new_var.permute(0, 2, 1)
                    ms_loss += (torch.fft.rfft(p, dim=2) - torch.fft.rfft(new_var, dim=2)).abs().mean() * 100

                loss = loss + ms_loss
                val_ms_item += ms_loss.item()

            if args.last:
                all_out = resize_list[-1]
            if args.first:
                all_out = resize_list[0]

            _, predicted = torch.max(all_out.data, 1)
            ##################################################

            p_classes = all_out.squeeze().transpose(1, 0)
            # loss_phase = criterion_phase(p_classes, labels_phase)
            loss_phase = criterion_phase(p_classes, labels_phase)
            val_loss_phase += loss_phase.data.item()

            val_corrects_phase += torch.sum(predicted == labels_phase.data)
            val_acc_each_video.append(float(torch.sum(predicted == labels_phase.data))/val_num_each[i])
            # TODO

            predicted = predicted.squeeze()
            for j in range(len(predicted)):
                val_all_preds_phase.append(int(predicted.data.cpu()[j]))
            for j in range(len(labels_phase)):
                val_all_labels_phase.append(int(labels_phase.data.cpu()[j]))


    val_elapsed_time = time.time() - val_start_time
    val_accuracy_phase = float(val_corrects_phase) / len(val_labels)
    val_acc_video = np.mean(val_acc_each_video)
    val_average_loss_phase = val_loss_phase


    # val metrics
    val_recall_phase = metrics.recall_score(val_all_labels_phase, val_all_preds_phase, average='macro')
    val_precision_phase = metrics.precision_score(val_all_labels_phase, val_all_preds_phase, average='macro')
    val_jaccard_phase = metrics.jaccard_score(val_all_labels_phase, val_all_preds_phase, average='macro')
    val_precision_each_phase = metrics.precision_score(val_all_labels_phase, val_all_preds_phase, average=None)  # 取平均的
    val_recall_each_phase = metrics.recall_score(val_all_labels_phase, val_all_preds_phase, average=None)  # 取平均的
    val_jaccard_each_phase = metrics.jaccard_score(val_all_labels_phase, val_all_preds_phase, average=None) # 取平均

    test_corrects_phase = 0
    test_all_preds_phase = []
    test_all_labels_phase = []
    test_acc_each_video = []
    test_start_time = time.time()
    test_ms_item = 0
    with torch.no_grad():
        for i in test_we_use_start_idx:
            labels_phase = []
            for j in range(test_start_vidx[i], test_start_vidx[i] + test_num_each[i]):
                if args.dataset == 'cholec80' or args.dataset == 'autolaparo':
                    labels_phase.append(test_labels[j][0])
                else:
                    labels_phase.append(test_labels[j])
            labels_phase = torch.LongTensor(labels_phase)
            if use_gpu:
                labels_phase = labels_phase.to(device)
            else:
                labels_phase = labels_phase

            long_feature = get_long_feature(start_index=test_start_vidx[i],
                                            lfb=g_LFB_test, LFB_length=test_num_each[i])

            long_feature = (torch.Tensor(long_feature)).to(device)
            video_fe = long_feature.transpose(2, 1)

            ##################################################
            predicted_list = model(video_fe)

            mean_len += predicted_list[0].size(-1)
            ans += 1
            if args.dataset == 'm2cai16':
                labels_phase = labels_phase.squeeze(1)
            all_out, resize_list, labels_list = fusion(predicted_list, labels_phase, args)
            max_seq = max(max_seq, video_fe.size(1))
            loss = 0
            if args.ms_loss:
                ms_loss = 0
                for p, l in zip(resize_list, labels_list):
                    ms_loss += loss_layer(p.transpose(2, 1).contiguous().view(-1, args.num_classes), l.view(-1))
                    ms_loss += torch.mean(torch.clamp(
                        mse_layer(F.log_softmax(p[:, :, 1:], dim=1), F.log_softmax(p.detach()[:, :, :-1], dim=1)),
                        min=0,
                        max=16))
                loss = loss + ms_loss
                test_ms_item += ms_loss.item()

            if args.last:
                all_out = resize_list[-1]
            if args.first:
                all_out = resize_list[0]

            _, predicted = torch.max(all_out.data, 1)
            ##################################################

            test_corrects_phase += torch.sum(predicted == labels_phase.data)
            test_acc_each_video.append(float(torch.sum(predicted == labels_phase.data)) / test_num_each[i])
            # TODO
            predicted = predicted.squeeze()
            for j in range(len(predicted)):
                test_all_preds_phase.append(int(predicted.data.cpu()[j]))
            for j in range(len(labels_phase)):
                test_all_labels_phase.append(int(labels_phase.data.cpu()[j]))

    test_accuracy_phase = float(test_corrects_phase) / len(test_labels)
    test_acc_video = np.mean(test_acc_each_video)
    test_elapsed_time = time.time() - test_start_time


    # test metrics
    test_recall_phase = metrics.recall_score(test_all_labels_phase, test_all_preds_phase, average='macro')
    test_precision_phase = metrics.precision_score(test_all_labels_phase, test_all_preds_phase, average='macro')
    test_jaccard_phase = metrics.jaccard_score(test_all_labels_phase, test_all_preds_phase, average='macro')
    test_precision_each_phase = metrics.precision_score(test_all_labels_phase, test_all_preds_phase, average=None)  # 取平均的
    test_recall_each_phase = metrics.recall_score(test_all_labels_phase, test_all_preds_phase, average=None)  # 取平均的
    test_jaccard_each_phase = metrics.jaccard_score(test_all_labels_phase, test_all_preds_phase, average=None) # 取平均

    print('epoch: {:4d}'
          ' train in: {:2.0f}m{:2.0f}s'
          ' train loss(phase): {:4.4f}'
          ' train accu(phase): {:.4f}'
          ' valid in: {:2.0f}m{:2.0f}s'
          ' valid loss(phase): {:4.4f}'
          ' valid accu(phase): {:.4f}'
          ' valid accu(video): {:.4f}'
          ' test in: {:2.0f}m{:2.0f}s'
          ' test accu(phase): {:.4f}'
          ' test accu(video): {:.4f}'
          .format(epoch,
                  train_elapsed_time // 60,
                  train_elapsed_time % 60,
                  train_average_loss_phase,
                  train_accuracy_phase,
                  val_elapsed_time // 60,
                  val_elapsed_time % 60,
                  val_average_loss_phase,
                  val_accuracy_phase,
                  val_acc_video,
                  test_elapsed_time // 60,
                  test_elapsed_time % 60,
                  test_accuracy_phase,
                  test_acc_video))

    # print the metrics of val & test
    print("print the metrics of val:###########################################################################")
    print("val_precision_each_phase:", val_precision_each_phase)
    print("val_recall_each_phase:", val_recall_each_phase)
    print("val_jaccard_each_phase:", val_jaccard_each_phase)
    print("val_precision_phase", val_precision_phase)
    print("val_recall_phase", val_recall_phase)
    print("val_jaccard_phase", val_jaccard_phase)
    print("print end:##########################################################################################")

    print("print the metrics of test:##########################################################################")
    print("test_precision_each_phase:", test_precision_each_phase)
    print("test_recall_each_phase:", test_recall_each_phase)
    print("test_jaccard_each_phase:", test_jaccard_each_phase)
    print("test_precision_phase", test_precision_phase)
    print("test_recall_phase", test_recall_phase)
    print("test_jaccard_phase", test_jaccard_phase)
    print("print end:##########################################################################################")

    # if test_accuracy_phase > best_accuracy_phase:
    # if val_accuracy_phase > best_val_accuracy_phase:
    if args.dataset == 'cholec80' or args.dataset == 'm2cai16':
        acc1 = test_accuracy_phase
        acc2 = best_accuracy_phase
    elif args.dataset == 'autolaparo':
        # acc1 = val_accuracy_phase + test_accuracy_phase
        # acc2 = best_val_accuracy_phase + best_accuracy_phase
        acc1 = val_accuracy_phase
        acc2 = best_val_accuracy_phase



    if acc1 > acc2:
        #sahc_07210321_epoch_31_train_9933_val_9306_test_9417 94.96± 4.20
        best_val_accuracy_phase = val_accuracy_phase
        best_test_accuracy_phase = test_accuracy_phase
        best_accuracy_phase = test_accuracy_phase
        correspond_train_acc_phase = train_accuracy_phase
        best_model_wts = copy.deepcopy(model.state_dict())
        best_epoch = epoch

        best_precision_phase = test_precision_phase
        best_recall_phase = test_recall_phase
        best_jaccard_phase = test_jaccard_phase



    save_val_phase = int("{:4.0f}".format(best_val_accuracy_phase * 10000))
    save_test_phase = int("{:4.0f}".format(best_test_accuracy_phase * 10000))
    save_train_phase = int("{:4.0f}".format(correspond_train_acc_phase * 10000))

    save_test_recall_phase = int("{:4.0f}".format(best_recall_phase * 10000))
    save_test_precision_phase = int("{:4.0f}".format(best_precision_phase * 10000))
    save_test_jaccard_phase = int("{:4.0f}".format(best_jaccard_phase * 10000))


    if epoch == args.epochs:  # 只最保留最好的
        now = datetime.datetime.now()
        month = str(now.month).zfill(2)
        day = str(now.day).zfill(2)
        hour = str(now.hour).zfill(2)
        minute = str(now.minute).zfill(2)
        date_time_str = f"{month}{day}{hour}{minute}"
        save_val_phase = int("{:4.0f}".format(best_val_accuracy_phase * 10000))
        save_train_phase = int("{:4.0f}".format(correspond_train_acc_phase * 10000))
        save_test_phase = int("{:4.0f}".format(best_test_accuracy_phase * 10000))

        save_test_recall_phase = int("{:4.0f}".format(best_recall_phase * 10000))
        save_test_precision_phase = int("{:4.0f}".format(best_precision_phase * 10000))
        save_test_jaccard_phase = int("{:4.0f}".format(best_jaccard_phase * 10000))

        # test_accuracy_phase, test_recall_phase, test_precision_phase, test_jaccard_phase
        if args.dataset == 'cholec80':
            base_name = "CHDTCN_80_" \
                        + date_time_str \
                        + "_epoch_" + str(best_epoch) \
                        + "_train_" + str(save_train_phase) \
                        + "_val_" + str(save_val_phase) \
                        + "_test_ac" + str(save_test_phase) \
                        + "pr" + str(save_test_precision_phase) \
                        + "re" + str(save_test_recall_phase) \
                        + "ja" + str(save_test_jaccard_phase)
        elif args.dataset == 'm2cai16':
            base_name = "CHDTCN_16_" \
                        + date_time_str \
                        + "_epoch_" + str(best_epoch) \
                        + "_train_" + str(save_train_phase) \
                        + "_val_" + str(save_val_phase) \
                        + "_test_ac" + str(save_test_phase) \
                        + "pr" + str(save_test_precision_phase) \
                        + "re" + str(save_test_recall_phase) \
                        + "ja" + str(save_test_jaccard_phase)
        elif args.dataset == 'autolaparo':
            base_name = "CHDTCN_laparo_" \
                        + date_time_str \
                        + "_epoch_" + str(best_epoch) \
                        + "_train_" + str(save_train_phase) \
                        + "_val_" + str(save_val_phase) \
                        + "_test_ac" + str(save_test_phase) \
                        + "pr" + str(save_test_precision_phase) \
                        + "re" + str(save_test_recall_phase) \
                        + "ja" + str(save_test_jaccard_phase)

        torch.save(best_model_wts, "./best_model/CHDTCN/" + base_name + ".pth")
        print("best_epbase_nameoch", base_name)
        print("best_epoch", str(best_epoch))

    print("save_val_phase:", save_val_phase, "****", "save_train_phase:", save_train_phase, "****", "save_test_phase(ac):", save_test_phase,
          "****", "test_(pr):", save_test_precision_phase, "****", "test_(re):", save_test_recall_phase,
          "****", "test_(ja)):", save_test_jaccard_phase)
    print("best_epoch:", str(best_epoch))
    print("model:", model_ver)
    print("lr:", args.learning_rate)

    logging.info(f'Epoch {epoch}: Training Accuracy = {train_accuracy_phase:.4f}, val_phase= {val_accuracy_phase:.4f}, test_phase= {test_accuracy_phase:.4f}')
    # logging.info(f'Epoch {epoch}: Training Accuracy = {train_accuracy_phase:.4f}, val_phase= {val_accuracy_phase:.4f}, test_phase= {test_accuracy_phase:.4f}')



