import openpyxl
import json
import pandas as pd
import sqlite3

# =========================
# データ加工
# =========================


df = pd.read_csv("model_race.csv", encoding='cp932')
print("データ数",len(df))

# 使用項目の絞込
use_col_name_list = []
one_hot_encoding_col_name_list = []
ordinal_col_name_list = []
flag_col_name_list = []
wb = openpyxl.load_workbook("データ項目.xlsx")
ws = wb["項目定義"]
for row in ws.iter_rows(min_row=2):  # 1行目はヘッダーなのでスキップ
    col_name = row[1].value
    # 使用項目
    if row[2].value == "○":
        # 項目名
        use_col_name_list.append(col_name)
        # 名義
        if row[3].value == "○":  
            one_hot_encoding_col_name_list.append(col_name)
        # 順序
        if row[4].value == "○":  
            ordinal_col_name_list.append(col_name)
        # フラグ
        if row[5].value == "○":  
            flag_col_name_list.append(col_name)

df = df[df["クラス名"]!="新馬"]

df = df.loc[:, use_col_name_list]

df = df.fillna(0)

arrival_result_normal_value_list = ["１","２","３","４","５","６","７","８","９","10","11","12","13","14","15","16","17","18","19","20"]

df = df[df["着順"].isin(arrival_result_normal_value_list)]
print("削除後データ数",len(df))


def convert_race_number(race_number):
    return str(race_number).zfill(2)  # レース番号を0埋めして2桁にそろえる
df["レース識別用コード"] = df["日付"].astype('str') + df["場所"] + df["Ｒ"].apply(convert_race_number)

df = df.drop(columns='日付')


categories_dict = {}
for one_hot_encoding_col_name in one_hot_encoding_col_name_list:
    # 各項目に対し一意の要素名を保持しておく（新規予測用）
    categories = list(set(df[one_hot_encoding_col_name].tolist()))
    categories_dict[one_hot_encoding_col_name] = categories
    # カテゴリ型に変換してからone-hot-encoding
    df[one_hot_encoding_col_name] = pd.Categorical(df[one_hot_encoding_col_name] , categories)
    df = pd.get_dummies(df, columns=[one_hot_encoding_col_name], dtype = int)


with open("categories_dict.dict", mode='wt', encoding='utf-8') as f:
    json.dump(categories_dict, f, ensure_ascii=False, indent=2)


for ordinal_col_name in ordinal_col_name_list:
    # 各変数名に対応した変換を行う
    # キー0は元々空白（Nan）だったデータ
    if ordinal_col_name in ["クラス名","前クラス名"]:
        map_dict = {"(不明)":0,"新馬":1,"未勝利":2,"1勝":3,"2勝":4,"3勝":5,"ｵｰﾌﾟﾝ":6,"OP(L)":7,
                    "Ｇ３":8,"Ｇ２":9,"Ｇ１":10,"重賞":8,"ＪＧ３":8,"ＪＧ２":9,"ＪＧ１":10,0:0}
    elif ordinal_col_name in ["馬場状態","前走馬場状態"]: 
        map_dict = {"良":1,"稍":2,"重":3,"不":4,0:0}
    elif ordinal_col_name in ["脚質"]:
        map_dict = {"逃げ":1,"先行":2,"中団":3,"後方":4,0:0}
    df[ordinal_col_name] = df[ordinal_col_name].map(map_dict)


for flag_col_name in flag_col_name_list:
    # 各変数名について「1」に変換するための対応する値を指定する
    if flag_col_name == "性別限定":
        target_value = "牝"
    elif flag_col_name in ["ブリンカー","前走B"]:
        target_value = "B"
    elif flag_col_name == "替":
        target_value = "*"
    elif flag_col_name == "芝(内・外)":
        # 半角スペースが入り込んでいるので完全一致させる
        target_value = " 外"

    def convert_values(value):
        if value == target_value:
            return 1
        else:
            return 0
    
    df[flag_col_name] = df[flag_col_name].apply(convert_values)


def extract_numbers(value):
    # ここではあらかじめ除去する記号が分かっているので直接指定していますが
    # 正規表現を用いて「整数か小数値」以外の文字を除去しても良いです。
    # また、「減量フラグ」という新たなデータ列を作ることもアリでしょう。
    return_value = value
    for remove_symbol in ["☆","◇","▲","△","★"]:
        return_value = return_value.replace(remove_symbol,"")
    return return_value 
for target_col_name in ["斤量","前走斤量"]:
    # 文字列に変換
    df[target_col_name] = df[target_col_name].astype('str')
    # 数字以外の記号を除去
    df[target_col_name] = df[target_col_name].apply(extract_numbers)


def convert_values(value):
    if value in arrival_result_normal_value_list:
        return int(value)
    else:
        return 0  # 正常着順以外はすべて0に置換
df["前走着順"] = df["前走着順"].apply(convert_values)  


df = df.replace("----", 0) 

conn = sqlite3.connect("modeling.db")

df.to_sql(
    "race_data",
    conn,
    if_exists="replace",
    index=False
)

conn.close()