from KG_loader import KG_Loader
from flashtext.keyword import KeywordProcessor
import random

# Part 1: Topology Neighbors

def buildEntLinkDict():
    # key: ent. value: [ head_dict(rel:t_nei), tail_dict(rel:h_nei) ]
    entityLinkDict = {}
    for triple in dm.trainList:
        h, t, r = triple
        if h not in entityLinkDict.keys():
            entityLinkDict[h] = [{}, {}]
        if r not in entityLinkDict[h][0].keys():
            entityLinkDict[h][0][r] = [t]
        else:
            entityLinkDict[h][0][r].append(t)

        if t not in entityLinkDict.keys():
            entityLinkDict[t] = [{}, {}]
        if r not in entityLinkDict[t][1].keys():
            entityLinkDict[t][1][r] = [h]
        else:
            entityLinkDict[t][1][r].append(h)
    return entityLinkDict

def getTopologyNeighbors(entLinkDict):
    relEntDict = {}
    for ent in entLinkDict.keys():
        relEntSet = set()
        tailDict, headDict = entLinkDict[ent]
        flag = False
        for item in tailDict.items():
            # print(ent, item[0], len(item[1]))
            rel = id2relationDict[item[0]]
            if len(item[1]) <= 3:
                relEntSet.update([rel + "||" + id2entityDict[e] for e in item[1]])
            else:
                relEntSet.update([rel + "||" + id2entityDict[e] for e in item[1][:3]])
        for item in headDict.items():
            # print(ent, item[0], len(item[1]))
            rel = id2relationDict[item[0]] + "_reverse"
            if len(item[1]) <= 3:
                relEntSet.update([rel + "||" + id2entityDict[e] for e in item[1]])
            else:
                relEntSet.update([rel + "||" + id2entityDict[e] for e in item[1][:3]])
        relEntDict[id2entityDict[ent]] = list(relEntSet)
    return relEntDict

# Part 2: Semantic Neighbors

def loadDescription(file, sp="\t"):
    entTextDict = {}
    with open(file, encoding="utf8") as file:
        lines = file.readlines()
        for line in lines:
            items = line.strip().split(sp)
            mid = items[0]
            text = items[-1]
            wordsStr = text.replace("-"," ").replace("-"," ")
            entTextDict[mid] = wordsStr
    return entTextDict

def loadName(file, sp="\t"):
    dict = {}
    with open(file, encoding="utf8") as file:
        lines = file.readlines()
        for line in lines:
            mid, name = line.strip().split(sp)
            dict[mid] = name.lower().replace("_"," ").replace("-"," ")
    return dict

def getSemanticNeighbors2(entNameDict, entTextDict):
    semNeiDict = {i: set() for i in dm.entityDict.keys()}
    num = 0
    for mid in entNameDict.keys():
        if num % 100 == 0:
            print(round(num/len(entNameDict),2))
        if mid not in dm.entityDict.keys(): continue
        name = entNameDict[mid]
        if len(name) <= 3: continue
        name = name.lower().replace("_", " ").replace("-", " ")
        for mid2 in entTextDict.keys():
            if mid2 not in dm.entityDict.keys(): continue
            if mid != mid2:
                text = entTextDict[mid2].replace("_", " ").replace("-", " ")
                #if name + " " in text or " "+ name in text:
                if " " + name + " " in text:
                    semNeiDict[mid].add(mid2)
                    semNeiDict[mid2].add(mid)
        num += 1
    return semNeiDict

def getSemanticNeighbors(entNameDict, entTextDict):
    semNeiDict = {i: set() for i in dm.entityDict.keys()}
    keyword_processor = KeywordProcessor()
    name2midDict = {}
    for mid in dm.entityDict.keys():
        if mid not in entNameDict.keys(): continue
        name = entNameDict[mid]
        if len(name) <= 3: continue
        keyword_processor.add_keyword(name)
        name2midDict[name] = mid

    for mid2 in entTextDict.keys():
        if mid2 not in dm.entityDict.keys(): continue
        text = entTextDict[mid2].lower()
        keywords_found = keyword_processor.extract_keywords(text)
        for key in keywords_found:
            mid = name2midDict[key]
            if mid != mid2:
                semNeiDict[mid].add(mid2)
                semNeiDict[mid2].add(mid)
    return semNeiDict

# Part 3: Composite Neighbors
def buildCompositeNeighbors(topNeiDict, semNeiDict):
    # the simplest strategy, bothNei + random_otherNei
    entityNeiEntDict = {}
    entityNeiRelDict = {}
    count, zeroNum, sumNum = 0, 0, 0
    entityNum = len(dm.entityDict)
    topNeiDictOutput, semNeiDictOutput = {}, {}
    for id in range(0, entityNum):
        id = id2entityDict[id]
        topNei = set([item.split("||")[1] for item in topNeiDict[id]]) if id in topNeiDict.keys() else set()
        semNei = set(semNeiDict[id]) if id in semNeiDict.keys() else set()
        if id in topNeiDict.keys():
            nei_relDict = {item.split("||")[1]: item.split("||")[0] for item in topNeiDict[id]}
        for nei in semNei:
            if nei not in nei_relDict.keys():
                nei_relDict[nei] = "text_semantic_related"

        bothNei = topNei & semNei
        otherNei = list((topNei | semNei) - bothNei)
        random.shuffle(otherNei)
        entityNeiEntDict[id] = list(bothNei) + otherNei
        entityNeiRelDict[id] = [nei_relDict[item] for item in entityNeiEntDict[id]]
        topNeiDictOutput[id] = list(topNei)
        semNeiDictOutput[id] = list(semNei)
        # print(id, len(bothNei), len(otherNei), len(topNei), len(semNei))
        bothLength = len(bothNei)
        count += 1
        sumNum += bothLength
        if bothLength == 0: zeroNum += 1
    print(len(entityNeiEntDict), "zeroNum", zeroNum, "meanNum", sumNum / count)
    return entityNeiEntDict, entityNeiRelDict, topNeiDictOutput, semNeiDictOutput

def saveNeighborFile(neiDict, filename):
    with open(filename, "w") as file:
        for mid, neis in neiDict.items():
            if mid == "null": continue
            file.write(mid + "\t" + " ".join(neis) + "\n")

def loadNeighborFile(filename):
    neiDict = {}
    with open(filename, "r") as file:
        for line in file.readlines():
            mid, neis = line.replace("\n","").split("\t")
            neiList = neis.split(" ")
        neiDict[mid] = neiList
    return neiDict

if __name__ == "__main__":
    path = "..\\data\\WN18RR\\"
    #path = "..\\data\\FB15k\\"
    dm = KG_Loader(path)
    dm.__extend__() # 扩展convE格式数据
    id2entityDict = dict(zip(dm.entityDict.values(), dm.entityDict.keys()))
    id2relationDict = dict(zip(dm.relationDict.values(), dm.relationDict.keys()))

    # Part 1: Topology Neighbors
    entLinkDict = buildEntLinkDict()
    topNeiDict = getTopologyNeighbors(entLinkDict)
    print(len(topNeiDict))

    # Part 2: Semantic Neighbors
    entNameDict = loadName(path + "entity_name.txt")
    entTextDict = loadDescription(path + "entity_description.txt")
    semNeiDict = getSemanticNeighbors(entNameDict, entTextDict)
    print(len(semNeiDict))

    # Part 3: Composite Neighbors containing both aboce parts
    entityNeiEntDict, entityNeiRelDict, topNeiDictOutput, semNeiDictOutput = buildCompositeNeighbors(topNeiDict, semNeiDict)
    saveNeighborFile(entityNeiEntDict, path + "comNeighbors_ent.txt")
    saveNeighborFile(entityNeiRelDict, path + "comNeighbors_rel.txt")
    saveNeighborFile(topNeiDictOutput, path + "topNeighbors_ent.txt")
    saveNeighborFile(semNeiDictOutput, path + "semNeighbors_ent.txt")
