import json
from collections import Counter
import zhconv

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

def contextIsInDocument(context, document):
    # return boolean
    # context: 要查詢的連續詞，list類型，包含多個字串
    # document: 文檔，list類型，包含多個字串
    if "".join(document).find("".join(list(context))) != -1:
        return True
    return False

def getAllContext(data):
    # return list
    # data: 文檔，2 layer list類型
    contextList = []
    for contextLength in range(1, 5): # n-gram 長度4
        print("n-gram:", contextLength)
        for document in data:
            for i in range(len(document) - contextLength + 1):
                l = document[i:i+contextLength]
                if l not in contextList:
                    contextList.append(l)
    return contextList

def calculateDF(data):
    # return dict
    # data: 所有文檔，2 layer list類型
    print("getting all context...")
    allContext = getAllContext(data)
    counter = Counter()
    contextTotal = len(allContext)
    print("calculate DF...")
    n = 1
    for context in allContext:
        print(n, "/", contextTotal)
        n += 1
        for i in data:
            if contextIsInDocument(context, i):
                counter.update([tuple(context)])
    return dict(counter)

def writeFile(DF, fileLocation):
    # DF: 每個context的DF值，dict類型
    # fileLocation: 寫入的檔案位置，string類型
    with open(fileLocation, "w", encoding="utf-8") as file:
        for i in DF:
            s = " ".join(i) + " " + str(DF[i])
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
    writeFile(DF, "DF.txt")

    print("Done")

main()