import numpy as np
import json
from collections import Counter
import zhconv
from tqdm import tqdm

def readJsonlFile(jsonlFile):
    # return 2 layer list
    # jsonlFile: 要讀入的檔案名稱，字串類型
    data = []
    with open(jsonlFile, "r") as file:
        for line in file:
            json_obj = json.loads(line)
            data.append(json_obj["zh"]["caption/tokenized/lowercase"])
    return data

def sentenceConnect(data):
    # return list
    # data: 要連接的句子，2 layer list類型
    document = []
    for i in data:
        s = i[0]
        for j in range(1, len(i)):
            s = s + " " + i[j]
        document.append(s)
    return document

def convertTW_and_split(data):
    # return 2 layer list
    # data: 要處理的句子，list類型，包含多個字串
    for i in range(len(data)):
        data[i] = zhconv.convert(data[i], "zh-tw")
        data[i] = list(data[i].split())
    return data

def getAllContext(data):
    # return list
    # data: 文檔，2 layer list類型
    contextSet = set() # 使用集合來存儲上下文
    for contextLength in range(1, 5): # n-gram 長度4
        for document in data:
            # 使用列表推導來生成所有的n-gram上下文
            contextSet.update([tuple(document[i:i+contextLength]) for i in range(len(document) - contextLength + 1)])
                    
    return list(contextSet) # 將集合轉換為列表返回

def calculateDF(data):
    # return dict
    # data: 所有文檔，2 layer list類型
    print("getting all context...")
    allContext = getAllContext(data)
    counter = Counter()
    print("calculate DF...")
    data_str = []
    for item in data:
        data_str.append("".join(item))
    for context in tqdm(allContext):
        context_str = "".join(list(context))
        for item in data_str:
            if item.find(context_str) != -1:
                counter.update([context])
    return dict(counter)

def writeFile(DF, fileLocation, totalDocsLen):
    # DF: 每個context的DF值，dict類型
    # fileLocation: 寫入的檔案位置，string類型
    with open(fileLocation, "w", encoding="utf-8") as file:
        for i in DF:
            s = " ".join(i) + " " + str(np.log(totalDocsLen / DF[i]))
            file.write(s)
            file.write("\n")

def main():
    # 文件處理
    print("File reading...")
    data = convertTW_and_split(sentenceConnect(readJsonlFile("captions.jsonl")))
    
    # 計算DF
    DF = calculateDF(data)

    #寫入檔案
    print("write file...")
    writeFile(DF, "DF.txt", len(data))

    print("Done")

main()