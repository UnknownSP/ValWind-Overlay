# ValWind-Overlay(開発停止中)
[VALORANT](https://playvalorant.com/ja-jp/)の公式大会のようにチームの体力などの情報をオーバーレイするためのプログラムになります。

以下動作例です。

https://github.com/UnknownSP/ValWind-Overlay/assets/39638661/2c6a607f-8383-4682-b47d-9ee13e8dd76f

## 仕様

OpenCVのmatchTemplateを用いて、あらかじめ用意してあるエージェントの画像と一致するかどうかで生存判断を行っています。
また、体力はHP部分の画像の色の差から判断しています。

## 課題と問題点

- ゲーム画面上部の背景色によって精度が落ちることがあり、不安定
- 最悪の場合不正ツールと判断される可能性がある
  - 公式からカスタムマッチのAPIが追加リリースされ次第利用
