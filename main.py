# 導入所需的套件
import numpy as np
from collections import Counter
from ckiptagger import WS
import json
import pygsheets

# 定義一個函數來計算 TF-IDF 值
def compute_tf_idf(cand, ref, n, mode, df):
    # cand: 候選描述，字串類型
    # ref: 參考描述，列表類型，包含多個字串
    # n: n-gram 的長度，整數類型
    # mode: IDF 的計算模式，字串類型，可以是 "corpus" 或 "coco-val-df"
    # df: IDF 的字典，字典類型，鍵為 n-gram，值為 IDF 值

    # 傳入分詞資料
    ws = WS("./data")

    # 將候選描述和參考描述進行分詞
    cand = ws([cand])[0]
    ref = ws(ref)
    # print(cand)
    
    # 將候選描述和參考描述轉換為 n-gram 的計數器
    cand_counter = Counter([tuple(cand[i:i+n]) for i in range(len(cand)-n+1)])
    ref_counter = [Counter([tuple(r[i:i+n]) for i in range(len(r)-n+1)]) for r in ref]

    # 計算候選描述的 TF 值
    cand_tf = {k: v / len(cand) for k, v in cand_counter.items()}

    # 計算參考描述的 TF 值
    ref_tf = [{k: v / len(r) for k, v in rc.items()} for rc, r in zip(ref_counter, ref)]

    # 計算候選描述和參考描述的 TF-IDF 值
    cand_tfidf = {}
    ref_tfidf = [{} for _ in range(len(ref))]
    for k in cand_tf.keys():
        if mode == "corpus":
            # 如果 IDF 模式是 "corpus"，則根據參考描述的出現次數計算 IDF 值
            idf = np.log((len(ref) + 1.0) / (1.0 + sum([1.0 for r in ref if k in r])))
        elif mode == "coco-val-df":
            # 如果 IDF 模式是 "coco-val-df"，則直接從 df 字典中獲取 IDF 值
            idf = df.get(k, 0.0)
        else:
            raise ValueError("Invalid mode: {}".format(mode))
        # 根據 TF 和 IDF 計算 TF-IDF 值
        cand_tfidf[k] = cand_tf[k] * idf
        for i in range(len(ref)):
            ref_tfidf[i][k] = ref_tf[i].get(k, 0.0) * idf
    
    # print(cand_tfidf, ref_tfidf, sep='\n------------------------\n')
    return cand_tfidf, ref_tfidf

# 定義一個函數來計算 CIDEr-D 分數
def compute_cider_d(cand, ref, n=4, mode="corpus", df=None):
    # cand: 候選描述，字串類型
    # ref: 參考描述，列表類型，包含多個字串
    # n: 最大的 n-gram 長度，整數類型
    # mode: IDF 的計算模式，字串類型，可以是 "corpus" 或 "coco-val-df"
    # df: IDF 的字典，字典類型，鍵為 n-gram，值為 IDF 值

    # 初始化 CIDEr-D 分數
    cider_d = 0.0

    # 對每個 n-gram 長度，計算 TF-IDF 值和餘弦相似度
    for i in range(1, n+1):
        # 計算候選描述和參考描述的 TF-IDF 值
        cand_tfidf, ref_tfidf = compute_tf_idf(cand, ref, i, mode, df)

        # 將 TF-IDF 值轉換為向量
        cand_vec = np.array([v for k, v in sorted(cand_tfidf.items())])
        ref_vec = np.array([[v for k, v in sorted(rt.items())] for rt in ref_tfidf])
        # print(cand_vec, ref_vec, sep='\n------------------------\n')

        # 計算候選描述和參考描述的餘弦相似度
        # cos_sim = np.dot(cand_vec, ref_vec.T) / (np.linalg.norm(cand_vec) * np.linalg.norm(ref_vec, axis=1))
        cos_sim = np.dot(cand_vec, ref_vec.T) / (np.linalg.norm(cand_vec) * np.linalg.norm(ref_vec, axis=1) + 1e-8)

        # 取餘弦相似度的最大值作為該 n-gram 的分數
        score = np.max(cos_sim)

        # 將該 n-gram 的分數累加到 CIDEr-D 分數中，並乘以一個權重因子 10.0 ** (i - 1)
        # cider_d += score * 10.0 ** (i - 1)
        cider_d += score / n
        
    return cider_d

def main():
    
    with open('setting.json') as f:
        data = dict(json.load(f).items())
    
    auth_file = data['auth_file']
    gc = pygsheets.authorize(service_file = auth_file)

    # setting sheet
    sheet_url = f"https://docs.google.com/spreadsheets/d/{data['sheet_id']}/"
    sheet = gc.open_by_url(sheet_url)
    
    sheet_description = sheet.worksheet_by_title(data['description_worksheet'])
    
    name = list(filter(None, sheet_description.get_col(1, include_tailing_empty=False)[1:]))
    blip_2 = list(filter(None, sheet_description.get_col(3, include_tailing_empty=False)[1:]))
    vit_gpt2 = list(filter(None, sheet_description.get_col(4, include_tailing_empty=False)[1:]))
    git = list(filter(None, sheet_description.get_col(5, include_tailing_empty=False)[1:]))
    ref = zip(list(filter(None, sheet_description.get_col(6, include_tailing_empty=False)[1:])), list(filter(None, sheet_description.get_col(7, include_tailing_empty=False)[1:])))

    count = 1
    result = {}
    for name, descriptions, ref in zip(name, zip(blip_2, vit_gpt2, git), ref):
        print(f"[INFO] Handling description {count}...")
        count += 1
        temp = []
        for cand in descriptions[:3]:
            temp.append(compute_cider_d(cand=cand, ref=ref))
        result[name] = temp
    
    print(f"[INFO] Updating google sheet...")
        
    sheet_cider = sheet.worksheet_by_title(data['cider_worksheet'])
    
    title = data['cider_worksheet_title']
    sheet_cider.update_row(1, values=title)
    
    row = 2
    for (name, ciders) in result.items():
        ciders.insert(0, name)
        sheet_cider.update_row(row, values=ciders)
        row += 1
        
    print(f"[INFO] Done.")
    
main()