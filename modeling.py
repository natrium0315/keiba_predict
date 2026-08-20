import pickle
import joblib
import pandas as pd
import numpy as np
import sqlite3
from lightgbm import LGBMClassifier

conn = sqlite3.connect("modeling.db")


df = pd.read_sql(
    "SELECT * FROM feature_data",
    conn
)

# df_numeric = df.select_dtypes(include=[np.number])

# df_numeric["レース識別用コード"] = df["レース識別用コード"]

# df = df_numeric

race_id_code_list = df["レース識別用コード"].unique().tolist()
race_id_code_list.sort()

train_data_count = int(2/3 * len(race_id_code_list))
train_race_id_code_list = race_id_code_list[:train_data_count]
test_race_id_code_list = race_id_code_list[train_data_count:]

df_train = df[df["レース識別用コード"].isin(train_race_id_code_list)].copy().sample(frac=1)
df_test = df[df["レース識別用コード"].isin(test_race_id_code_list)].copy()



column_name_list_input =  df.columns.to_list()  

column_name_list_input.remove("Ｒ")
column_name_list_input.remove("馬名")
column_name_list_input.remove("勝利")
column_name_list_input.remove("着順")
column_name_list_input.remove("3着以内")
column_name_list_input.remove("レース識別用コード")  
column_name_list_input.remove("単勝オッズ")
column_name_list_input.remove("単勝配当")
column_name_list_input.remove("複勝配当")

joblib.dump(column_name_list_input, "columns.pkl")

output_symbol = "3着以内"
#output_symbol = "勝利"
column_name_list_output = [output_symbol]

np_train_input = df_train.loc[:, column_name_list_input].values  # 学習データ-説明変数  
np_train_output = df_train.loc[:, column_name_list_output].values  # 学習データ-目的変数
np_test_input = df_test.loc[:, column_name_list_input].values # テストデータ-説明変数
np_test_output = df_test.loc[:, column_name_list_output].values # テストデータ-目的変数



model = LGBMClassifier(
    n_estimators=1000,
    learning_rate=0.05,
    max_depth=8,
    random_state=42
)

model.fit(np_train_input, np_train_output[:, 0])

pickle.dump(model, open("lgbm.model", 'wb'))



predict_val = model.predict_proba(np_test_input) 

df_test["予測値"] = predict_val[:,1]  # 予測値を追加
tmp_count = 0   # 予測データ数（本命馬数）カウント用
tmp_hit = 0  # 複勝（単勝）的中数カウント用
tmp_ret = 0  # 複勝（単勝）配当合計金額計算用
df_test["本命"] = 0  # 本命馬行が分かるようにフラグ用列用意（PG上では使わないが後ほどデータ確認するため）
# テスト用レースをそれぞれループ
for test_race_id_code in test_race_id_code_list:
    # 該当レースに出走した馬のデータのみ抽出
    df_pick = df_test[df_test["レース識別用コード"]==test_race_id_code]
    # 各レースについて予測値最大となった馬のデータを抽出
    df_pick_max = df_pick.loc[df_pick["予測値"].idxmax()]
    # 本命馬にフラグ立て
    df_test.loc[df_pick["予測値"].idxmax(),"本命"] = 1
    if df_pick_max["予測値"] > 0:
        tmp_count += 1
        if output_symbol == "勝利":  # 単勝成績算出用
            # 本命馬が的中しているか？
            if df_pick_max["勝利"] == 1:
                # 単勝的中！
                tmp_hit += 1  # 的中数加算
                tmp_ret += int(df_pick_max["単勝配当"])  # 配当加算
        elif output_symbol == "3着以内":  # 複勝成績算出用
            if df_pick_max["3着以内"] == 1:
                # 複勝的中！
                tmp_hit += 1
                tmp_ret += int(df_pick_max["複勝配当"])
    

if output_symbol == "勝利": 
    print("単勝成績")
elif output_symbol == "3着以内":
    print("複勝成績")
print("的中率 回収率 的中数 予測数 払戻合計")
print(tmp_hit/tmp_count,tmp_ret/(tmp_count*100),tmp_hit,tmp_count,tmp_ret)

conn.close()