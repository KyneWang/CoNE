import numpy as np
from numpy.random import choice
from collections import defaultdict
import math, random
import torch

class KG_DataSet(object):
    def __init__(self, xs, sampler, mode, config):
        self.xs = xs
        self.mode = mode
        self.sampler = sampler
        self.config = config
        self.data_size = len(xs)
        self.batch_size = 16
        if mode == "train":
            self.batch_size = config.batch_size
        elif mode == "test" or mode == "valid":
            self.batch_size = config.test_batch_size
        self.nbatches = math.ceil(self.data_size / float(self.batch_size))
        return

    def getFilter(self, triple, mode = "train"):
        return self.sampler.loader.getFilter(triple, mode)

    def getBatch(self):
        if self.mode == "train":
            rand_idx = np.random.permutation(self.data_size)
        else:
            rand_idx = range(self.data_size)
        start = 0
        while start < self.data_size:
            end = min(start + self.batch_size, self.data_size)
            if self.mode == "train":
                batch = [self.xs[i] for i in rand_idx[start:end]]
                ntsH, nstT = self.sampler.getBatch2(batch, n=self.config.nagTriple_num)  # get nagtive triples
                Sbatch = [batch, ntsH, nstT]
            else:
                Sbatch = [self.xs[i] for i in rand_idx[start:end]]
            yield Sbatch
            start = end

class DataSet_Builder(object):
    def __init__(self, loader, config, name=""):
        self.name = name or "dataset"
        self.config = config
        self.loader = loader
        self.sampler = KGSampler(loader)
        self.batch_size = config.batch_size

    def build(self):
        self.train_xs = self.loader.trainList
        self.valid_xs = self.loader.validList
        self.test_xs = self.loader.testList

        train_dataset = KG_DataSet(self.train_xs, self.sampler, "train", self.config)
        valid_dataset = KG_DataSet(self.valid_xs, self.sampler, "valid", self.config)
        test_dataset = KG_DataSet(self.test_xs, self.sampler, "test", self.config)

        return train_dataset, valid_dataset, test_dataset

# 后续考虑，如何选择和目标实体有一定语义关联的负样本
class KGSampler(object):
    def __init__(self, loader):
        self.loader = loader
        self.bern_prob = self.get_bern_prob(loader.trainList, loader.entityNum, loader.relationNum)
        self.n_ent = loader.entityNum

    def getBatch0(self, batch, n = 1):
        src, dst, rel = zip(*batch)
        src = np.array(src, dtype="int64")
        dst = np.array(dst, dtype="int64")
        prob = self.bern_prob[list(rel)]
        total_List = []
        for _ in range(n):
            #selection = torch.bernoulli(prob).numpy().astype('int64')
            ent_random = list(range(self.n_ent))[:len(src)]
            src_out = ent_random
            ent_random = list(range(self.n_ent))[:len(dst)]
            dst_out = ent_random
            total_List.append([src_out, dst_out])
        #     del (src_out, dst_out, selection)
        # del(src, dst, prob)
        total_List = np.array(total_List).transpose([1,2,0]).tolist()
        return total_List #(2, batch_size, neg_num)

    def getBatch(self, batch, n = 1):
        src, dst, rel = zip(*batch)
        src = np.array(src, dtype="int64")
        dst = np.array(dst, dtype="int64")
        prob = self.bern_prob[list(rel)]
        total_List = []
        for _ in range(n):
            selection = torch.bernoulli(prob).numpy().astype('int64')
            ent_random = choice(self.n_ent, len(src))
            src_out = (1 - selection) * src + selection * ent_random
            dst_out = selection * dst + (1 - selection) * ent_random
            total_List.append([src_out, dst_out])
        #     del (src_out, dst_out, selection)
        # del(src, dst, prob)
        total_List = np.array(total_List).transpose([1,2,0]).tolist()
        return total_List #(2, batch_size, neg_num)

    # 后续考虑增加bern, 用于RotatE
    def getBatch2(self, batch, n = 1):
        ntriples_batch, ntriples_batch2 = [], []
        for triple in batch:
            trueT, trueH = self.loader.getFilter(triple)
            negH = self.__sampleNagtiveTriples(list(trueH), n)
            negT = self.__sampleNagtiveTriples(list(trueT), n)
            # negH = choice(self.n_ent, n)
            # negT = choice(self.n_ent, n)
            ntriples_batch.append(negH)
            ntriples_batch2.append(negT)
        return ntriples_batch, ntriples_batch2

    def __sampleNagtiveTriples2(self, trueList, n=1):
        n_list = random.sample(list(set(range(self.n_ent)) - set(trueList)), n)
        negative_sample = np.array(n_list)[:n]
        return negative_sample

    def __sampleNagtiveTriples(self, trueList, n=1):
        n_list = []
        n_size = 0
        while n_size < n:
            negative_sample = np.random.randint(self.n_ent, size=n * 2)
            mask = np.in1d(negative_sample,trueList,assume_unique=True,invert=True)
            negative_sample = negative_sample[mask]
            n_list.append(negative_sample)
            n_size += negative_sample.size
        negative_sample = np.concatenate(n_list)[:n]
        return negative_sample

    def get_bern_prob(self, data, n_ent, n_rel):
        src, dst, rel = zip(*data)
        edges = defaultdict(lambda: defaultdict(lambda: set()))
        rev_edges = defaultdict(lambda: defaultdict(lambda: set()))
        for s, r, t in zip(src, rel, dst):
            edges[r][s].add(t)
            rev_edges[r][t].add(s)
        bern_prob = torch.zeros(n_rel)
        for r in edges.keys():
            tph = sum(len(tails) for tails in edges[r].values()) / len(edges[r])
            htp = sum(len(heads) for heads in rev_edges[r].values()) / len(rev_edges[r])
            bern_prob[r] = tph / (tph + htp)
        return bern_prob

