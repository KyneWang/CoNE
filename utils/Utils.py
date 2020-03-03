import torch
import numpy as np
from scipy.stats import rankdata

# basic method reading data from file
def loadFile(path):
    with open(path, "r", encoding="utf8") as df:
        data = df.readlines()
    return data

# basic method writing data into file
def saveFile(path, dataList):
    with open(path, "w", encoding="utf8") as df:
        for line in dataList:
            df.write(line+"\n")

# get file's line number quickly
def getFileLength(path):
     return len(["" for l in open(path, "r", encoding="utf8")])

def loadNpy(path):
    return np.load(path)

def saveNpy(data, path):
    np.save(path, data)
    print("saved", path)

# metrics
def mrr_mr_hitk(scores, target, k=10):
    scores = torch.sigmoid(scores)
    sorted_vec, sorted_idx = torch.sort(scores)
    find_target = sorted_idx == target
    target_rank = torch.nonzero(find_target)[0, 0] + 1
    #print(sorted_vec[:10], target_rank)
    metrics = [float(1 / target_rank), float(target_rank), int(target_rank <= k), int(target_rank <= 3), int(target_rank <= 1)]
    return np.array(metrics, dtype=np.float32)

def mrr_mr_hitk_show(scores, target, k=10):
    #scores2 = torch.sigmoid(scores)
    # sorted_idx = rankdata(scores.data.cpu().numpy(), method="average")
    # target_rank1 = sorted_idx[target]
    # sorted_idx = rankdata(scores.data.cpu().numpy(), method = "ordinal")
    # target_rank2 = sorted_idx[target]
    # sorted_idx = rankdata(scores.data.cpu().numpy(), method = "min")
    # target_rank3 = sorted_idx[target]
    # sorted_idx = rankdata(scores.data.cpu().numpy(), method="max")
    # target_rank4 = sorted_idx[target]
    sorted_vec, sorted_idx = torch.sort(scores)
    #sorted_idx = torch.from_numpy(np.argsort(scores.data.cpu().numpy()))
    find_target = sorted_idx == target
    target_rank = torch.nonzero(find_target)
    target_rank = target_rank[0, 0] + 1 if len(target_rank) > 0 else len(sorted_idx)
    # print(target_rank1, target_rank2, target_rank3, target_rank4)
    # print(target_rank)
    # target_rank = target_rank3
    metrics = [float(1 / target_rank), float(target_rank), int(target_rank <= k), int(target_rank <= 3), int(target_rank <= 1)]
    return np.array(metrics, dtype=np.float32),sorted_vec[:k], target_rank, sum(metrics[2:])

def mrr_mr_hitk_new(scores, target, method = "average", k=10):
    # "average"  "ordinal" "min" "max"
    sorted_vec, _ = torch.sort(scores)
    sorted_idx = rankdata(scores.data.cpu().numpy(), method).astype(int)
    target_rank = sorted_idx[target] if target < len(sorted_idx) else len(sorted_idx)
    metrics = [float(1 / target_rank), float(target_rank), int(target_rank <= k), int(target_rank <= 3), int(target_rank <= 1)]
    return np.array(metrics, dtype=np.float32), sorted_vec[:k], target_rank, sum(metrics[2:])

def mrr_mr_hitk_show2(scores, target, k=10):
    #sorted_idx = rankdata(scores)
    #target_rank = sorted_idx[target]
    sorted_vec, sorted_idx = torch.sort(scores, descending=True)
    find_target = sorted_idx == target
    target_rank = torch.nonzero(find_target)
    target_mr = float(target_rank[0, 0] + 1) if len(target_rank) > 0 else float(len(sorted_idx))
    target_mrr = float(1 /(target_rank[0, 0] + 1)) if len(target_rank) > 0 else float(0)
    metrics = [target_mrr, target_mr, int(target_mr <= k), int(target_mr <= 3), int(target_mr <= 1)]
    return np.array(metrics, dtype=np.float32),sorted_vec[:k], target_rank, sum(metrics[2:])

def init_norm_Vector(relinit, entinit, embedding_size):
    zero_vec = [0.1 for i in range(embedding_size)]
    lstent = [zero_vec]
    lstrel = [zero_vec]
    with open(relinit) as f:
        for line in f:
            tmp = [float(val) for val in line.strip().split()]
            lstrel.append(tmp)
    with open(entinit) as f:
        for line in f:
            tmp = [float(val) for val in line.strip().split()]
            lstent.append(tmp)
    assert embedding_size % len(lstent[0]) == 0
    return np.array(lstent, dtype=np.float32), np.array(lstrel, dtype=np.float32)


class Regularization(torch.nn.Module):
    def __init__(self, model, weight_decay, p=2):
        '''
        :param model 模型
        :param weight_decay:正则化参数
        :param p: 范数计算中的幂指数值，默认求2范数,
                  当p=0为L2正则化,p=1为L1正则化
        '''
        super(Regularization, self).__init__()
        if weight_decay <= 0:
            print("param weight_decay can not <=0")
            exit(0)
        self.model = model
        self.weight_decay = weight_decay
        self.p = p
        self.weight_list = self.get_weight(model)
        self.weight_info(self.weight_list)

    # def to(self, device):
    #     '''
    #     指定运行模式
    #     :param device: cude or cpu
    #     :return:
    #     '''
    #     self.device = device
    #     super().to(device)
    #     return self

    def forward(self, model):
        self.weight_list = self.get_weight(model)  # 获得最新的权重
        reg_loss = self.regularization_loss(self.weight_list, self.weight_decay, p=self.p)
        return reg_loss

    def get_weight(self, model):
        '''
        获得模型的权重列表
        :param model:
        :return:
        '''
        weight_list = []
        for name, param in model.named_parameters():
            if 'weight' in name:
                weight = (name, param)
                weight_list.append(weight)
        return weight_list

    def regularization_loss(self, weight_list, weight_decay, p=2):
        '''
        计算张量范数
        :param weight_list:
        :param p: 范数计算中的幂指数值，默认求2范数
        :param weight_decay:
        :return:
        '''
        # weight_decay=Variable(torch.FloatTensor([weight_decay]).to(self.device),requires_grad=True)
        # reg_loss=Variable(torch.FloatTensor([0.]).to(self.device),requires_grad=True)
        # weight_decay=torch.FloatTensor([weight_decay]).to(self.device)
        # reg_loss=torch.FloatTensor([0.]).to(self.device)
        reg_loss = 0
        for name, w in weight_list:
            l2_reg = torch.norm(w, p=p)
            reg_loss = reg_loss + l2_reg

        reg_loss = weight_decay * reg_loss
        return reg_loss

    def weight_info(self, weight_list):
        '''
        打印权重列表信息
        :param weight_list:
        :return:
        '''
        print("---------------regularization weight---------------")
        for name, w in weight_list:
            print(name)
        print("---------------------------------------------------")
