# circle_classifier

## 概要
2次元座標が円の内側にあるか外側にあるかを判別する2値分類器を、3層のパーセプトロンとして実装した。

人工ニューロン、誤差逆伝搬法、交差エントロピー損失といった講義内容の理解を目的としている。

## 問題設定
入力は2次元座標とし、原点を中心とする半径rの円の内側にある場合を1、外側にある場合を0とした。

この問題は線形分離不可能であり、単層のパーセプトロンでは解けない。

## 使用したモデル
以下の構成の多層パーセプトロンを用いた。

<img width="50%" src="https://github.com/YazawaKenichi/circle_classification/blob/main/img/nn.png">

出力層にシグモイド関数を用いることで、出力を「円の内側である確率」として解釈できる。

## 学習方法
損失関数には、2値分類の交差エントロピー損失を用いた。

$$
L = - ( t log(y) + (1 - t) log(1 - y))
$$

ここで、tは正解のラベル、yはモデルの出力である。

## 実験結果
<img src = "https://github.com/YazawaKenichi/circle_classification/blob/main/img/result-plot.png">
学習後の確率の勾配。黄色が円内の確率が高く、紫色が円外の確率が高い。

<img src = "https://github.com/YazawaKenichi/circle_classification/blob/main/img/loss-graph.png">
エポックごとの平均損失。

|入力座標|(0, 0)|(0.6, 0.75)|(0.6, 0.75)
|:---|:---|:---|:---
|epoch|100|500|500
|alpha|0.001|0.001|0.001
|教師データ数|1000|1000|10000
|最終的な損失|0.0849|0.0015|1.17e-05
|円内の確率|0.749|0.345|0.989

epoch = 100
alpha = 0.001
教師データ数 = 1000
からこの学習をはじめた。

入力された座標に対するモデルの出力を図として確認した結果、
円の内外に対応した出力が得られていることが分かった。

境界に近い座標(0.6, 0.25)では、35% と低い確率であった
教師データの数を1000から10000に変更して学習したところ確率は98.9%まで上がった


また、epoch の増加に伴って損失が減少していくことを確認できた。

## 実行環境
- Ubuntu 22.04
- Python3.10
で動作確認済み。

## 実行方法
### クローン
``` bash
git clone https://github.com/YazawaKenichi/circle_classification
cd circle_classification
```

### 必要なライブラリのインストール
- numpy
- argparse
- matplotlib

``` bash
pip install -r requirements.txt
```

### 実行
``` bash
./demo.sh
```

## 参考
- [ryuichiueda/slides_marp - GitHub](https://github.com/ryuichiueda/slides_marp/tree/master/advanced_vision)
- [基本的な出力層の活性化関数と損失関数の組み合わせまとめ](https://qiita.com/pocokhc/items/d67b63ec9ca74b453093)
- [【ディープラーニング入門（第5回）】勾配降下法を学んでディープラーニングの学習について理解しよう - Qiita](https://qiita.com/kwi0303/items/e43efa6657ff4b6b59ad)
- [深層学習／活性化関数たち](https://qiita.com/jun40vn/items/2ca86ab6b821ae20086c)

