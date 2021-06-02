# M-LSD-warpPerspective-Example
[M-LSD](https://github.com/navervision/mlsd)を用いて四角形を検出し、射影変換を行うサンプルプログラムです。<br><br>
<img src="https://user-images.githubusercontent.com/37477845/120508690-e4a61380-c402-11eb-9f7a-0cc4eadc9e53.gif" width="50%">

# Requirements
* OpenCV 3.4.2 or Later
* tensorflow 2.4.1 or Later

# Usage
実行方法は以下です。<br>
```bash
python example.py
```
<br>
実行時には、以下のオプションが指定可能です。

* --device<br>
カメラデバイス番号の指定<br>
デフォルト：0
* --file<br>
動画ファイルの指定 ※指定時にはカメラデバイスより優先<br>
デフォルト：None
* --width<br>
カメラキャプチャ時の横幅<br>
デフォルト：640
* --height<br>
カメラキャプチャ時の縦幅<br>
デフォルト：480
* --crop_width<br>
射影変換後の画像の横幅<br>
デフォルト：224
* --crop_height<br>
射影変換後の画像の縦幅<br>
デフォルト：224
* --model<br>
モデルパス<br>
デフォルト：mlsd/tflite_models/M-LSD_320_tiny_fp32.tflite
* --model_shape<br>
モデルの入力形状幅<br>
デフォルト：320
* --top_n<br>
検出スコアの高い順にいくつ使用するか<br>
デフォルト：1
* --score<br>
M_LSDパラメータ：score<br>
デフォルト：0.1
* --outside_ratio<br>
M_LSDパラメータ：outside_ratio<br>
デフォルト：0.1
* --inside_ratio<br>
M_LSDパラメータ：inside_ratio<br>
デフォルト：0.5
* --w_overlap<br>
M_LSDパラメータ：w_overlap<br>
デフォルト：0.0
* --w_degree<br>
M_LSDパラメータ：w_degree<br>
デフォルト：1.14
* --w_length<br>
M_LSDパラメータ：w_length<br>
デフォルト：0.03
* --w_area<br>
M_LSDパラメータ：w_area<br>
デフォルト：1.84
* --w_center<br>
M_LSDパラメータ：w_center<br>
デフォルト：1.46

# Reference
推論用プログラム([mlsd/utils.py](mlsd/utils.py))、および学習済モデル([mlsd/tflite_models](mlsd/tflite_models))は、<br>
[navervision/mlsd](https://github.com/navervision/mlsd)リポジトリのものを使用しています。

# Author
高橋かずひと(https://twitter.com/KzhtTkhs)
 
# License 
M-LSD-warpPerspective-Example is under [Apache v2 license](LICENSE).

