import pickle
import joblib
import pandas as pd
import numpy as np
import sqlite3
from lightgbm import LGBMClassifier

conn = sqlite3.connect("modeling.db")

race_id_code_list = pd.read_sql(
    """
    SELECT DISTINCT レース識別用コード
    FROM feature_data
    """,
    conn
)["レース識別用コード"].tolist()

race_id_code_list.sort()

train_data_count = int(9/10 * len(race_id_code_list))
train_race_id_set = set(
    race_id_code_list[:train_data_count]
)
test_race_id_set = set(
    race_id_code_list[train_data_count:]
)

columns = pd.read_sql(
    "SELECT * FROM feature_data LIMIT 1",
    conn
).columns.tolist()


column_name_list_input = columns.copy()

column_name_list_input.remove("Ｒ")
column_name_list_input.remove("馬名")
column_name_list_input.remove("勝利")
column_name_list_input.remove("着順")
column_name_list_input.remove("3着以内")
column_name_list_input.remove("レース識別用コード")
column_name_list_input.remove("単勝オッズ")
column_name_list_input.remove("単勝配当")
column_name_list_input.remove("複勝配当")

joblib.dump(
    column_name_list_input,
    "columns.pkl"
)

#output_symbol = "3着以内"
output_symbol = "勝利"
column_name_list_output = [output_symbol]


# =========================
# 学習データ
# =========================
def train():
    conn = sqlite3.connect("modeling.db")
    train_input_list = []
    train_output_list = []

    train_count = 0

    for df in pd.read_sql(
        "SELECT * FROM feature_data",
        conn,
        chunksize=5000
    ):
        train_count += len(df)
        print(f'学習データ処理中: {train_count} まで')
        df = df[
            df["レース識別用コード"].isin(train_race_id_set)
        ]

        if len(df) == 0:
            del df
            continue

        train_input_list.append(
            df[column_name_list_input].to_numpy(dtype = np.float32)
        )

        train_output_list.append(
            df[output_symbol].to_numpy()
        )

        del df

    np_train_input = np.concatenate(
        train_input_list,
        axis=0
    )

    np_train_output = np.concatenate(
        train_output_list,
        axis=0
    )

    del train_input_list
    del train_output_list
    conn.close()
    return np_train_input, np_train_output

# =========================
# テストデータ
# =========================
def test():
    conn = sqlite3.connect("modeling.db")
    test_data_list = []

    test_count = 0

    for df in pd.read_sql(
        "SELECT * FROM feature_data",
        conn,
        chunksize=5000
    ):
        test_count += len(df)
        print(f'テストデータ処理中: {test_count} まで')
        df = df[
            df["レース識別用コード"].isin(test_race_id_set)
        ]

        if len(df) == 0:
            del df
            continue

        test_data_list.append(df)

        del df

    df_test = pd.concat(
        test_data_list,
        ignore_index=True
    )

    del test_data_list


    np_test_input = df_test[
        column_name_list_input
    ].to_numpy()

    conn.close()
    return np_test_input, df_test

# =========================
# モデル作成
# =========================
def modeling(np_train_input, np_train_output):
    model = LGBMClassifier(
        n_estimators=1000,
        learning_rate=0.05,
        max_depth=8,
        random_state=42
    )

    model.fit(np_train_input, np_train_output)

    pickle.dump(model, open("solo_lgbm.model", 'wb'))

    return model


def evaluate(model, np_test_input, df_test):
    predict_val = model.predict_proba(np_test_input) 

    df_test["予測値"] = predict_val[:,1]  # 予測値を追加
    tmp_count = 0   # 予測データ数（本命馬数）カウント用
    tmp_hit = 0  # 複勝（単勝）的中数カウント用
    tmp_ret = 0  # 複勝（単勝）配当合計金額計算用
    df_test["本命"] = 0  # 本命馬行が分かるようにフラグ用列用意（PG上では使わないが後ほどデータ確認するため）
    # テスト用レースをそれぞれループ
    for test_race_id_code in test_race_id_set:
        # 該当レースに出走した馬のデータのみ抽出
        df_pick = df_test[df_test["レース識別用コード"]==test_race_id_code]
        # 各レースについて予測値最大となった馬のデータを抽出
        df_pick_max = df_pick.loc[df_pick["予測値"].idxmax()]
        # 本命馬にフラグ立て
        df_test.loc[df_pick["予測値"].idxmax(),"本命"] = 1
        if df_pick_max["予測値"]:
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
    
    return tmp_count, tmp_hit, tmp_ret
    
def main():
    train_input, train_output = train()
    model = modeling(train_input, train_output)
    del train_input, train_output
    test_input, df_test = test()
    tmp_count, tmp_hit, tmp_ret = evaluate(model, test_input, df_test)
    if output_symbol == "勝利": 
        print("単勝成績")
    elif output_symbol == "3着以内":
        print("複勝成績")
    print("的中率 回収率 的中数 予測数 払戻合計")
    print(tmp_hit/tmp_count,tmp_ret/(tmp_count*100),tmp_hit,tmp_count,tmp_ret)

conn.close()

if __name__ == "__main__":
    main()