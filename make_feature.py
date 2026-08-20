import pandas as pd
import sqlite3

conn = sqlite3.connect("modeling.db")

df = pd.read_sql(
    "SELECT * FROM race_data",
    conn
)

print("データ数:", len(df))

# 数値化
df["斤量"] = pd.to_numeric(df["斤量"], errors="coerce")
df["前走斤量"] = pd.to_numeric(df["前走斤量"], errors="coerce")

df["距離"] = pd.to_numeric(df["距離"], errors="coerce")
df["前距離"] = pd.to_numeric(df["前距離"], errors="coerce")

# =========================
# 特徴量作成
# =========================
df = df.sort_values(["レース識別用コード", "単勝オッズ"])

df["距離変化量"] = (df["距離"] - df["前距離"]).where(df["前距離"] > 0)

df["斤量変化量"] = (df["斤量"] - df["前走斤量"]).where(df["前走斤量"] > 0)

df["ダート牝馬"] = ((df["芝・ダ_ダ"] == 1) & (df["性別_牝"] == 1)).astype(int)

df["ダートから芝"] = ((df["前芝・ダ_ダ"] == 1) & (df["芝・ダ_芝"] == 1)).astype(int)

df["昇格戦"] = (df["前クラス名"] < df["クラス名"]).astype(int)

same_race = df["レース識別用コード"] == df["レース識別用コード"].shift(-1)
df["オッズ比"] = (df["単勝オッズ"].shift(-1) / df["単勝オッズ"]).where(same_race)

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
    if_exists="replace",
    index=False
)

conn.close()