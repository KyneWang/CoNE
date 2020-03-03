import torch
from torch.autograd import Variable
import progressbar as pb
import numpy as np
import datetime, random, re
from utils.KG_loader import KG_Loader
from utils.Dataset_builder import DataSet_Builder, KG_DataSet
from utils.ExtraDataManager import ExtraDataManager
from utils.Utils import *
from model import CoNE_model

class Config(object):
    def __init__(self):
        self.dataset = "WN18RR" #"WN18RR" #"FB15k-237"
        self.data_path = "data\\"+self.dataset+"\\"
        self.save_path = "save\\"
        self.encoder_name = "GMN"
        self.decoder_name = "TransE"
        self.neighbor_name = "Com"
        self.trainFlag = True
        self.loadFlag = False
        self.cuda = True
        self.learning_rate = 0.0001
        self.embedding_dim = 100
        self.trainTimes = 500
        self.batch_size = 1000
        self.nei_size = 20 #20
        self.margin = 8
        self.nagTriple_num = 1
        self.test_batch_size = 32
        self.test_period = 2
        self.save_flag = False
        self.entityNum = 0
        self.relationNum = 0
        self.nbatches = 0

def main():
    config = Config()
    # load basic data
    loader = KG_Loader(config.data_path)
    config.entityNum = loader.entityNum
    config.relationNum = loader.relationNum
    config.startTimeSpan = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
    # load extra data
    dm = ExtraDataManager(config.data_path, loader.entityDict, loader.relationDict2)
    neiMatrix = np.array(dm.load_EntNei(path=config.data_path + "entity_neighbors_com.txt",max_length = 20))
    neiMatrix = neiMatrix[:,:config.nei_size]
    # build dataset
    builder = DataSet_Builder(loader, config)
    train_ds, valid_ds, test_ds = builder.build()

    eMat, rMat = None, None
    eMat = loadNpy(config.save_path + "WN18RR_TransE_entMatrix_e306_hit10@47.15_20190927_210621.npy")
    rMat = loadNpy(config.save_path + "WN18RR_TransE_relMatrix_e306_hit10@47.15_20190927_210621.npy")
    if config.decoder_name == "RotatE": # as the entity dim is two times of relation dim
        rMat = np.concatenate([rMat, rMat], axis=1)
        config.embedding_dim = config.embedding_dim * 2

    model = CoNE_model(config)
    model.init(neiMat=neiMatrix, eMat=eMat, rMat=rMat)
    #opt = torch.optim.Adam(model.parameters(), lr=config.learning_rate, weight_decay=0.000)
    opt = torch.optim.SGD(model.parameters(), lr=config.learning_rate, weight_decay=0.000)  # 237
    if config.cuda:
        model.cuda()

    for epoch_idx in range(config.trainTimes):
        model.train(mode=True)
        totalLoss = train(model, train_ds, opt, config, epoch_idx)
        print("Loss", totalLoss.item(), round(totalLoss.item() / train_ds.data_size, 5))
        with torch.no_grad():
            model.eval()
            if epoch_idx % config.test_period == 0:
                metric = eval(model, valid_ds, "valid", 100, config)
            if epoch_idx % 20 == 0:
                print("whole test")
                eval(model, test_ds, "test", test_ds.nbatches, config)

def matrixSave(save_dict, config, epoch, metric):
    print("saving", save_dict.keys())
    name = config.dataset + "_" + config.modelName + "_Nei_"
    for k, v in save_dict.items():
        np.save('{0}{1}_{2}_hit10@{3}_{4}.npy'.format(config.save_path + name, k, str(epoch), str(round(metric, 2)),
                                                      config.startTimeSpan), v)

def train(model, train_ds, opt, config, epoch_idx):
    pbar = pb.ProgressBar(widgets=["epoch %d|" % (epoch_idx + 1),
                                   pb.Percentage(), pb.Bar(), pb.ETA()], maxval=train_ds.nbatches)
    pbar.start()
    totalLoss, num_batches_completed = 0, 0
    for singlebatch in train_ds.getBatch():
        loss = pair_Train(model, singlebatch, opt, config)
        totalLoss += loss.data
        pbar.update(num_batches_completed)
        num_batches_completed += 1
    pbar.finish()
    return totalLoss

def pair_Train(model, singlebatch, opt, config):
    opt.zero_grad()
    batchO, batchH, batchT = singlebatch
    e1_batch, r_batch, e2_batch = [], [], []
    for item in batchO:
        e1, e2, r = item[:3]
        e1_batch.append([e1])
        r_batch.append([r])
        e2_batch.append([e2])

    e1_varb = Variable(torch.LongTensor(np.array(e1_batch, dtype=np.int32))).cuda()
    rel_varb = Variable(torch.LongTensor(np.array(r_batch, dtype=np.int32))).cuda()
    e2_varb = Variable(torch.LongTensor(np.array(e2_batch, dtype=np.int32))).cuda()

    e1n_varb = Variable(torch.LongTensor(np.array(batchH, dtype=np.int32))).cuda()
    e2n_varb = Variable(torch.LongTensor(np.array(batchT, dtype=np.int32))).cuda()

    lossH = torch.sum(model.loss(e1_varb, e2_varb, rel_varb, e1n_varb, e2_varb, "head_batch"))
    lossT = torch.sum(model.loss(e1_varb, e2_varb, rel_varb, e1_varb, e2n_varb, "tail_batch"))
    loss = lossH + lossT

    loss.backward()
    opt.step()
    model.constraint()

    del (e1_varb, rel_varb, e2_varb, e1n_varb,e2n_varb)
    return loss

def eval(model, valid_ds, mode, test_batch, config, sort_mode = "average"):
    pbar = pb.ProgressBar(widgets=["eval %s|" % mode, pb.Percentage(), pb.Bar(), pb.ETA()], maxval=test_batch)
    pbar.start()
    num_batches_completed = 0
    metricsVecH, metricsVecT = [], []
    for singlebatch in valid_ds.getBatch():
        e1, e2, rel, pred1, pred2 = pair_Eval(model, singlebatch, config)
        torch.cuda.empty_cache()
        batch_size = len(singlebatch)
        for i in range(batch_size):
            num1 = e1[i, 0].item()
            num2 = e2[i, 0].item()
            num_rel = rel[i, 0].item()
            filter1, filter2 = valid_ds.getFilter((num1, num2, num_rel), mode="eval")
            target_value1 = pred1[i, num2].item()
            target_value2 = pred2[i, num1].item()
            pred1[i][list(filter1)] = 1e35  # 0.0
            pred2[i][list(filter2)] = 1e35  # 0.0
            pred1[i][e2[i]] = target_value1
            pred2[i][e1[i]] = target_value2

            metricsH, topk_scores, predict_rank, predict_level = mrr_mr_hitk_new(pred1[i], num2, sort_mode)
            metricsT, topk_scores, predict_rank, predict_level = mrr_mr_hitk_new(pred2[i], num1, sort_mode)
            metricsVecH.append(metricsH)
            metricsVecT.append(metricsT)

        num_batches_completed += 1
        if num_batches_completed >= test_batch:
            break
        pbar.update(num_batches_completed)

    pbar.finish()

    hit10 = MetricsOutput(metricsVecH, metricsVecT, type="Filter")
    return hit10

def MetricsOutput(metricsVecH, metricsVecT, type):
    print("#"*10, type, "#"*10)
    mrr, mr, hit10, hit3, hit1 = np.array(metricsVecH).mean(axis=0)
    print('H MRR:', round(np.mean(1. / np.array(metricsVecH)[:,1]), 4), ' MR:', round(mr, 2),
          'Hit@10:', round(hit10 * 100, 2),
          'Hit@3:', round(hit3 * 100, 2),
          'Hit@1:', round(hit1 * 100, 2))

    mrr, mr, hit10, hit3, hit1 = np.array(metricsVecT).mean(axis=0)
    print('T MRR:', round(np.mean(1. / np.array(metricsVecT)[:,1]), 4), ' MR:', round(mr, 2),
          'Hit@10:', round(hit10 * 100, 2),
          'Hit@3:', round(hit3 * 100, 2),
          'Hit@1:', round(hit1 * 100, 2))

    mrr, mr, hit10, hit3, hit1 = np.array(metricsVecH+metricsVecT).mean(axis = 0)
    print('MRR:',round(np.mean(1. / np.array(metricsVecH+metricsVecT)[:,1]),4), ' MR:', round(mr,2),
          'Hit@10:',round(hit10 * 100,2),
          'Hit@3:',round(hit3 * 100,2),
          'Hit@1:', round(hit1*100,2))
    return round(hit10 * 100,2)

def pair_Eval(model, singlebatch, config):
    e1_batch, r_batch, e2_batch = [], [], []
    for item in singlebatch:
        e1, e2, r = item[:3]
        e1_batch.append(e1)
        r_batch.append(r)
        e2_batch.append(e2)
    batch_size = len(singlebatch)
    n_ent = config.entityNum
    batch_s = torch.LongTensor(np.array(e1_batch, dtype=np.int32))
    batch_r = torch.LongTensor(np.array(r_batch, dtype=np.int32))
    batch_t = torch.LongTensor(np.array(e2_batch, dtype=np.int32))
    rel_var = Variable(batch_r.unsqueeze(1).expand(batch_size, 1).cuda())
    src_var = Variable(batch_s.unsqueeze(1).expand(batch_size, 1).cuda())
    dst_var = Variable(batch_t.unsqueeze(1).expand(batch_size, 1).cuda())

    all_var = Variable(torch.arange(0, n_ent).unsqueeze(0).expand(batch_size, n_ent).type(torch.LongTensor).cuda())
    pred1 = model.score(src_var, rel_var, all_var, "tail_batch").data
    pred2 = model.score(all_var, rel_var, dst_var, "head_batch").data
    e1, e2, rel = src_var.data, dst_var.data, rel_var.data
    del all_var, rel_var, src_var, dst_var
    if config.decoder_name == "RotatE":
        pred1, pred2 = - pred1.data, - pred2.data
    if config.decoder_name == "ConvE":
        pred1, pred2 = (1 - pred1.data), (1 - pred2.data)
    return e1, e2, rel, pred1, pred2

if __name__ == '__main__':
    main()