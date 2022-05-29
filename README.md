<img src="README-IMG/header.png" width="1600mm">

# 【Kaggle】Tabular Playground Series - May 2022 

## 概要
**Tabular Playground Series - May 2022**のためのリポジトリ  
CSV形式で与えられた訓練データから何らかの手法を用いてtargetラベル(0~1)を予測する


## 入力データについて
Pandasを用いてDataFrame型で読み込む

```
import pandas as pd

train_path = "input/train.csv"
train = pd.read_csv(train_path)
```
データ内容を確認してみる。先頭から5行を出力。
```
print(train.head(5))
```
結果
```
   id      f_00      f_01      f_02  ...        f_28  f_29  f_30  target
0   0 -1.373246  0.238887 -0.243376  ...   67.609153     0     0       0
1   1  1.697021 -1.710322 -2.230332  ...  377.096415     0     0       1
2   2  1.681726  0.616746 -1.027689  ... -195.599702     0     2       1
3   3 -0.118172 -0.587835 -0.804638  ...  210.826205     0     0       1
4   4  1.148481 -0.176567 -0.664871  ... -217.211798     0     1       1
[5 rows x 33 columns]
```
32種類のの特徴量と1種類のクラスラベルを確認。  
次に型の確認。
```
train.info()
```
結果
```
RangeIndex: 900000 entries, 0 to 899999
Data columns (total 33 columns):
 #   Column  Non-Null Count   Dtype  
---  ------  --------------   -----  
 0   id      900000 non-null  int64  
 1   f_00    900000 non-null  float64
 2   f_01    900000 non-null  float64
 3   f_02    900000 non-null  float64
 4   f_03    900000 non-null  float64
 5   f_04    900000 non-null  float64
 6   f_05    900000 non-null  float64
 7   f_06    900000 non-null  float64
 8   f_07    900000 non-null  int64  
 9   f_08    900000 non-null  int64  
 10  f_09    900000 non-null  int64  
 11  f_10    900000 non-null  int64  
 12  f_11    900000 non-null  int64  
 13  f_12    900000 non-null  int64  
 14  f_13    900000 non-null  int64  
 15  f_14    900000 non-null  int64  
 16  f_15    900000 non-null  int64  
 17  f_16    900000 non-null  int64  
 18  f_17    900000 non-null  int64  
 19  f_18    900000 non-null  int64  
 20  f_19    900000 non-null  float64
 21  f_20    900000 non-null  float64
 22  f_21    900000 non-null  float64
 23  f_22    900000 non-null  float64
 24  f_23    900000 non-null  float64
 25  f_24    900000 non-null  float64
 26  f_25    900000 non-null  float64
 27  f_26    900000 non-null  float64
 28  f_27    900000 non-null  object 
 29  f_28    900000 non-null  float64
 30  f_29    900000 non-null  int64  
 31  f_30    900000 non-null  int64  
 32  target  900000 non-null  int64  
dtypes: float64(16), int64(16), object(1)
```
targetを除くと、訓練データは15種類のint型(64bit)データと16種類のfloat型(64bit)データと1種の文字列型で構成されていることがわかる。  
特徴量としてf_00~f30の値と、f_02とf_21、f_05とf_22、f_00とf_01とf_26の値の和が閾値を超えるか否か(bool)の34種を選択する。  
また、f_27はユニークな文字数(int)に変換して使用する。

## 学習と推論

`optuna_catboost.py`で最適なパラメータを推定  
↓  
得たパラメータを`validation_learn.py`で設定して学習、推論  
また、`tabular_learn.py`でvalidationをしない学習、推論
  
もしくは、`Tensorflow_Keras_learn.py`でCNNによる学習、推論を行う。

