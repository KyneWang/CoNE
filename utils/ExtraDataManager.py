# load extra data about KG from files
import numpy as np
import utils.Utils as Utils

class ExtraDataManager(object):
    def __init__(self, path, entityDict, relationDict):
        self.data_path = path
        self.entityDict = entityDict
        self.relationDict = relationDict

    def load_EntNei(self, path, max_length):
        ent2neiDict = self.load_entMetaInfo(path)
        ent2neiMatrix = np.zeros([len(self.entityDict), max_length])
        for key, value in ent2neiDict.items():
            neis = [self.entityDict[item] for item in value.split(" ") if item != ""]
            neis = neis[:max_length]
            neis = neis + [0 for i in range(max_length-len(neis))]
            ent2neiMatrix[int(key)] = np.array(neis)
        return ent2neiMatrix

    # load from entName/Type/Neighbor.txt
    def load_entMetaInfo(self, file, sp="\t"):
        dict = {id:"" for id in self.entityDict.values()}
        data = Utils.loadFile(file)
        for line in data:
            mid, item = line.replace("\n","").split(sp)
            if mid not in self.entityDict.keys(): continue
            dict[self.entityDict[mid]] = item
        return dict
