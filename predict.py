import openpyxl
import json
import pickle
import joblib
import pandas as pd
import numpy as np
from openpyxl import load_workbook
from lightgbm import LGBMClassifier



def pre_process_data(target_data_path):
    """
    新規予測用データの作成（加工） 準備
    新規予測用データを学習時のデータ項目に合わせる
    """
    print("1.モデル構築用データの作成（加工） 準備","開始")
    
    # オリジナルデータの読込
    df = pd.read_csv(target_data_path,
                        encoding='cp932'
                        )

    # 項目名一覧取得
    use_col_name_list = list(df.columns.values)
    # 使用しない項目を削除
    use_col_name_list.remove("種牡馬")
    use_col_name_list.remove("調教師")
    use_col_name_list.remove("同場")
    use_col_name_list.remove("前走開催")   
    # 使用項目の絞込
    df = df.loc[:, use_col_name_list]
    # print(df.head(5))

    # 「性別限定」列の生成
    def convert_race_name(race_name):
        # レース名に「牝」という文字が含まれていたら「牝」、それ以外は空白に置換
        if "牝" in race_name:
            return "牝"
        else:
            return ""
    df["性別限定"] = df["レース名"].apply(convert_race_name)

    # 「レース名」列は削除 
    df = df.drop(columns='レース名')

    # 特定列のデータが学習時と異なっているので置換（これを行わないとカテゴリ値変換で失敗する）


    # 項目名の置換（学習時のデータ項目名に合わせる）
    # キー：置換前項目名、値：置換後項目名 の辞書
    replace_column_name_dict = {
        "日付S":"日付",
        "クラス":"クラス名",
        "芝ダ":"芝・ダ",
        "外回り":"芝(内・外)",
        "頭":"頭数",
        "年限定":"年齢限定",
        "単オッズ":"単勝オッズ",
        "B":"ブリンカー",
        "前場所":"前走場所",
        "前ｸﾗｽ":"前クラス名",
        "前芝ダ":"前芝・ダ",
        "前馬場状態":"前走馬場状態",
        "前頭数":"前走頭数",
        "前番":"前走馬番",
        "乗替":"替",
        "前走騎手":"前騎手",
        "前馬体重":"前走馬体重",
        "前馬体重増減":"前走馬体重増減",
        "前斤量":"前走斤量",
        "前人気":"前走人気",
        "前確定着順":"前走着順",
        "前着差":"前走着差タイム",
        "前通過1":"前1角",
        "前通過2":"前2角",
        "前通過3":"前3角",
        "前通過4":"前4角",
        "前3F順":"前走上り3F順"
    }
    df = df.rename(columns=replace_column_name_dict)

    # キー：置換対象項目名、値：「置換前データと置換後データの辞書」 の辞書
    replace_column_data_dict =  {
        "年齢限定":{
        "４歳以上":"４上",
        "３歳以上":"３上"
        },
        "前走馬場状態":{
        "稍重":"稍",
        "不良":"不"
        },
        "替":{
        "替":"*",
        },
        "前走馬体重":{
        "---":0,
        },
        # 他にもありそうなら追記
    }
    df.replace(replace_column_data_dict,inplace=True)

    # CSV出力
    export_csv_name = "pre_proceed_data.csv"
    df.to_csv(export_csv_name,
                encoding='cp932',
                index=False)

    print("1.データ加工準備完了")
    
    return export_csv_name



def process_data(target_data_path):
    df = pd.read_csv(target_data_path, encoding='cp932')
    print("データ数",len(df))

    original_col_name_list = list(df.columns.values)

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
            if col_name in original_col_name_list:
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

    # ------後でやるとごちゃるので特例---------------------------
    df["乗り替わり"] = (df["騎手"] != df["前騎手"]).astype(int)
    # ---------------------------------------------------------

    def convert_race_number(race_number):
        return str(race_number).zfill(2)  # レース番号を0埋めして2桁にそろえる
    df["レース識別用コード"] = df["日付"].astype('str') + df["場所"] + df["Ｒ"].apply(convert_race_number)

    df = df.drop(columns='日付')

    with open("categories_dict.dict", mode='rt', encoding='utf-8') as f:
        categories_dict = json.load(f)

    for one_hot_encoding_col_name in one_hot_encoding_col_name_list:
        categories = categories_dict[one_hot_encoding_col_name]
        # カテゴリ型に変換してからone-hot-encoding
        df[one_hot_encoding_col_name] = pd.Categorical(df[one_hot_encoding_col_name] , categories)
        df = pd.get_dummies(df, columns=[one_hot_encoding_col_name], dtype = int)


    for ordinal_col_name in ordinal_col_name_list:
        # 各変数名に対応した変換を行う
        # キー0は元々空白（Nan）だったデータ
        if ordinal_col_name in ["クラス名","前クラス名"]:
            map_dict = {"(不明)":0,"新馬":1,"未勝利":2,"1勝":3,"2勝":4,"3勝":5,"ｵｰﾌﾟﾝ":6,"OP(L)":7,
                        "Ｇ３":8,"Ｇ２":9,"Ｇ１":10,"重賞":8,"ＪＧ３":8,"ＪＧ２":9,"ＪＧ１":10,0:0}
        elif ordinal_col_name in ["馬場状態","前走馬場状態"]: 
            map_dict = {"良":1,"稍":2,"重":3,"不":4,0:0}
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
        try:
            return int(value)
        except ValueError:
            return 0  # 正常着順以外はすべて0に置換
    df["前走着順"] = df["前走着順"].apply(convert_values)  


    df = df.replace("----", 0) 


    #df = df.drop(columns=['所属_[外]','前走場所_英国','前走場所_韓国','前走場所_ﾄﾞｲﾂ'])


    df.to_csv("proceed_data.csv", encoding='cp932', index=False)
    print("2.データ加工完了")



def make_feature_value():
    df = pd.read_csv("proceed_data.csv", encoding='cp932')

    df["距離変化量"] = (df["距離"] - df["前距離"]).where(df["前距離"] > 0)

    df["斤量変化量"] = (df["斤量"] - df["前走斤量"]).where(df["前走斤量"] > 0)

    df["ダート牝馬"] = ((df["芝・ダ_ダ"] == 1) & (df["性別_牝"] == 1)).astype(int)

    df["ダートから芝"] = ((df["前芝・ダ_ダ"] == 1) & (df["芝・ダ_芝"] == 1)).astype(int)

    df["昇格戦"] = (df["前クラス名"] < df["クラス名"]).astype(int)

    places = ["東京", "中山", "阪神", "京都", "中京",
          "小倉", "新潟", "福島", "札幌", "函館"]

    df["前走同場"] = 0

    for place in places:
        df.loc[
            (df[f"場所_{place}"] == 1) &
            (df[f"前走場所_{place}"] == 1),
            "前走同場"
        ] = 1
    
    df.to_csv("proceed_data2.csv", encoding='cp932', index=False)
    print("3.特徴量作成完了")



def predict():
    df = pd.read_csv("proceed_data2.csv", encoding='cp932')

    race_id_code_list = df["レース識別用コード"].unique().tolist()
    race_id_code_list.sort()

    train_columns = joblib.load("columns.pkl")

    X = df.reindex(columns=train_columns, fill_value=0)
    
    model = pickle.load(open("lgbm.model", "rb"))

    predict_val = model.predict_proba(X.values) 

    df["予測値"] = predict_val[:,1]

    with open("result.txt", "w", encoding="utf-8")as f:
        for race_id in race_id_code_list:
            race_df = df[df["レース識別用コード"] == race_id]

            top = race_df.sort_values("予測値", ascending=False)

            f.write(f"レース: {race_id}\n")
            for _, r in top.iterrows():
                f.write(
                    f'予測:{r["馬番"]}:{r["馬名"]}:{r["予測値"]}\n'
                )





# 予測対象データ（TARGET）
target_data_path = "race.csv"
# モデル構築用データの作成（加工） 準備
export_csv_name = pre_process_data(target_data_path)
# 予測用データの作成（加工）
process_data(export_csv_name)
# 簡単な特徴量の作成
make_feature_value()
# 新規予測＆結果まとめ
predict()