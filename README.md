# circle_classifier

## 概要
2次元座標が円の内側にあるか外側にあるかを判別する2値分類器を、3層のパーセプトロンとして実装した。

人工ニューロン、誤差逆伝搬法、交差エントロピー損失といった講義内容の理解を目的としている。

## 問題設定
入力は2次元座標とし、原点を中心とする半径rの円の内側にある場合を1、外側にある場合を0とした。

この問題は線形分離不可能であり、単層のパーセプトロンでは解けない。

## 使用したモデル
以下の構成の多層パーセプトロンを用いた。

**ここに図**

出力層にシグモイド関数を用いることで、出力を「円の内側である確率」として解釈できる。

## 学習方法
損失関数には、2値分類の交差エントロピー損失を用いた。[参考](https://qiita.com/pocokhc/items/d67b63ec9ca74b453093)

$$
L = - ( t log(y) + (1 - t) log(1 - y))
$$

ここで、tは正解のラベル、yはモデルの出力である。

## 実験結果
入力された座標に対するモデルの出力を図として確認した結果、
円の内外に対応した出力が得られていることが分かった。

<img src="https://github.com/YazawaKenichi/circle_classification/blob/main/img/1768110697-500-0.0005-prediction.png">

また、epoch の増加に伴って損失が減少していくことを確認できた。

<img src="https://github.com/YazawaKenichi/circle_classification/blob/main/img/1768110697-500-0.0005-loss.png">

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
- [Lorem Ipsum](https://example.com)

