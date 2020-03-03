# load KG data from files
import utils.Utils as Utils
import copy
import numpy as np

# main class
class KG_Loader(object):
    def __init__(self, path, ifprint = True):
        self.data_path = path
        self.ifprint = ifprint
        self.__load__()
        self.filter = KGFilter([self.trainList, self.validList, self.testList], self.entityNum)
        if ifprint:
            print("trainNum", self.trainNum, "validNum", self.validNum, "testNum", self.testNum)
            print("entityNum", self.entityNum, "relationNum", self.relationNum)

    # load from file
    def __load__(self):
        # load idDict
        self.entityNum, self.entityDict = self.load_Item2idDict(self.data_path + "entity2id.txt")
        self.relationNum, self.relationDict = self.load_Item2idDict(self.data_path + "relation2id.txt")

        # load triples
        self.trainList, self.trainNum = self.load_Triples(self.data_path + "train.txt")
        self.validList, self.validNum = self.load_Triples(self.data_path + "valid.txt")
        self.testList, self.testNum = self.load_Triples(self.data_path + "test.txt")
        self.wholeList = self.trainList + self.validList + self.testList

    def getFilter(self, triple, mode="train"):
        filter1, filter2 = self.filter.getFilter(triple, mode)
        return filter1, filter2

    # load from item2id.txt,
    def load_Item2idDict(self, file, sp="\t"):
        list = {}
        data = Utils.loadFile(file)
        list["null"] = 0
        for line in data:
            item, id = line.strip().split(sp)
            list[item] = int(id) + 1  # note: id+1, set 0 as null
        return len(list), list

    # load from train/test/valid.txt
    def load_Triples(self, file, rank = "htr", sp="\t"):
        list = []
        data = Utils.loadFile(file)
        for line in data:
            triple = line.strip().split(sp)
            if(len(triple)<3):
                continue
            if rank == "hrt":
                h, r, t = triple
            elif rank == "htr":
                h, t, r = triple
            if h not in self.entityDict.keys() or t not in self.entityDict.keys():
                continue
            if r not in self.relationDict.keys():
                continue
            list.append(tuple([self.entityDict[h],self.entityDict[t],self.relationDict[r]]))
        return list, len(list)

class KGFilter(object):
    def __init__(self, dataLists, entityNum):
        self.entityNum = entityNum
        self.trainList,self.validList,self.testList = dataLists
        self.wholeList = self.trainList.copy()
        self.wholeList.extend(self.validList)
        self.wholeList.extend(self.testList)
        self.train_dict = self.__build_filter(self.trainList)
        self.label_dict = self.__build_filter(self.wholeList)
        self.train_ent2relDict = self.__build_erdict(self.trainList)

    def getFilter(self, triple, mode = "train"):
        h, t, r = triple
        key = str(h) + ">" + str(r)
        key2 = str(r) + "<" + str(t)
        if mode == "train":
            pairDict = self.train_dict
        elif mode == "predicteval":
            set_train = set(self.train_dict[key]) if key in self.train_dict.keys() else set()
            set1 = set(self.label_dict[key]) - set_train
            set_train2 = set(self.train_dict[key2]) if key2 in self.train_dict.keys() else set()
            set2 = set(self.label_dict[key2]) - set_train2
            return set1, set2
        else:
            pairDict = self.label_dict
        h,t,r = triple
        key = str(h) + ">" + str(r)
        key2 = str(r) + "<" + str(t)
        #print(h, t, r, key, key2)
        return pairDict[key], pairDict[key2]

    def __build_filter(self, datalist):
        pairDict = {}
        for h,t,r in datalist:
            key = str(h) + ">" + str(r)
            if key not in pairDict.keys():
                pairDict[key] = [t]
            else:
                pairDict[key].append(t)
            key = str(r) + "<" + str(t)
            if key not in pairDict.keys():
                pairDict[key] = [h]
            else:
                pairDict[key].append(h)
        return pairDict

    def __build_erdict(self, datalist):
        pairDict = {}
        for h, t, r in datalist:
            if h not in pairDict.keys():
                pairDict[h] = []
            if t not in pairDict.keys():
                pairDict[t] = []
            pairDict[h].append(r)
            pairDict[t].append(-r)
        return pairDict


if __name__ == "__main__":
    path = "..\\data\\FB15k-237\\"
    dm = KG_Loader(path)
    dm.__extend__() # 扩展convE格式数据

