import pandas as pd
import sqlite3

# ========================= 
# DB接続 
# =========================
conn = sqlite3.connect("modeling.db")

cursor = conn.cursor()

data_count = cursor.execute(
    "SELECT COUNT(*) FROM race_data"
).fetchone()[0]

CHUNK_SIZE = 5000

first_chunk = True
l = 0

conn.execute("DROP TABLE IF EXISTS feature_data")
conn.commit()

for df in pd.read_sql(
    "SELECT * FROM race_data",
    conn,
    chunksize = CHUNK_SIZE
):
    l += len(df)
    p = int(l / data_count * 100)
    print(f'現在処理中: {p} %')

    # 数値化
    df["斤量"] = pd.to_numeric(df["斤量"], errors="coerce")
    df["前走斤量"] = pd.to_numeric(df["前走斤量"], errors="coerce")

    df["距離"] = pd.to_numeric(df["距離"], errors="coerce")
    df["前距離"] = pd.to_numeric(df["前距離"], errors="coerce")

    # =========================
    # 特徴量作成
    # =========================

    df["距離変化量"] = (df["距離"] - df["前距離"]).where(df["前距離"] > 0)

    df["斤量変化量"] = (df["斤量"] - df["前走斤量"]).where(df["前走斤量"] > 0)

    df["ダート牝馬"] = ((df["芝・ダ_ダ"] == 1) & (df["性別_牝"] == 1)).astype(int)

    df["ダートから芝"] = ((df["前芝・ダ_ダ"] == 1) & (df["芝・ダ_芝"] == 1)).astype(int)

    df["昇格戦"] = (df["前クラス名"] < df["クラス名"]).astype(int)

    df["勝利"] = (df["着順"] == "１").astype(int)

    df["3着以内"] = df["着順"].isin(["１","２","３"]).astype(int)


    places = ["東京", "中山", "阪神", "京都", "中京",
            "小倉", "新潟", "福島", "札幌", "函館"]

    df["前走同場"] = 0

    for place in places:
        df.loc[
            (df[f"場所_{place}"] == 1) &
            (df[f"前走場所_{place}"] == 1),
            "前走同場"
        ] = 1

    # =========================
    # DBへ保存
    # =========================

    df.to_sql(
        "feature_data",
        conn,
        if_exists="append",
        index=False
    )

    del df

conn.close()
print("特徴量作成完了")