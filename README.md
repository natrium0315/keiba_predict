<img width="1926" height="816" alt="de6ff36d-9c32-4271-bd9f-a56956707e18" src="https://github.com/user-attachments/assets/8d53151c-b230-42b4-a115-bdd50a065f6b" />

# 開発の動機
大学の友人に誘われて始めた競馬ですが、競馬場が家から近いこともあり、すっかりハマってしまいました。

そこで情報学部に所属している私は思いました。「競馬予測AI作れば楽勝じゃない？」と。
競馬の波に乗っていた私は、WEBを参考にAIを適宜用いながら一か月で予測AIとして動くものを完成させ、今も改善をつづけています。

# 概要
使用するデータは過去10年分のデータをJRA-VANデータラボのサービスであるTARGET frontier JVからcsvで抽出しています。

csvデータをまずmake_data.pyにて使用するデータを選び、データを加工します。

次にmake_feature.pyにて特徴量を作成、追加します。

最後にmodeling.py(ex_modeling.py)にて「学習用データ：テストデータ＝９：１」としてモデル作成とその評価を行います。

当日のレースの予測はpredict.pyが行います。その結果はresult.txtとして保存されます。

# 使用技術
| 分類 | 技術 |
|---|---|
| 言語 | Python |
| 機械学習 | LightGBM |
| データ処理 | Pandas, NumPy |
| データベース | SQLite |
| データ保存 | Pickle |
| バージョン管理 | Git, GitHub |

# 今後の展望
- 競馬予測AIとしての機能の充実、かつ予測精度の向上。
- JV-Linkからのデータ自動取得。
- Flaskとjinjaを用いたWEBアプリケーションとして動くように改善。
