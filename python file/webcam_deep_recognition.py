# -*- coding: utf-8 -*-
import keras
from keras import backend as K
import webcam_whiteblack as w_b
import webcam_getVector as getvc
import webcam_screenshot as getss
import cv2
import numpy as np
from scipy import stats
import sys

getss.getScreenshot()
# 手の二値化画像のサイズ 縦(row)と横(col)
img_rows, img_cols = 12, 16

# 学習に用いる縮小画像のサイズ
sw = img_cols
sh = img_rows

# 手の認識用パラメータ（HチャンネルとSチャンネルとを二値化するための条件）
hmin = 0
hmax = 30 # 15-40程度にセット
smin = 50

janken_class =  ['グー', 'パー']

# 学習済ファイルの確認
if len(sys.argv)==1:
    print('使用法: webcam_deep_recognition.py 学習済ファイル名.h5')
    sys.exit()
savefile = sys.argv[1]

# 学習済ファイルを読み込んでmodelを作成
model = keras.models.load_model(savefile)

print('認識を開始します')



            # 最大の白領域からkerasに入力するためのベクトルを取得
while True:
    hand = w_b.white_black()
    hand_vector = getvc.getImageVector(hand,sw,sh)
    X = np.array(hand_vector)
    if K.image_data_format() == 'channels_first':
        X = X.reshape(X.shape[0], 1, img_rows, img_cols)
        input_shape = (1, img_rows, img_cols)
    else:
        X = X.reshape(X.shape[0], img_rows, img_cols, 1)
        input_shape = (img_rows, img_cols, 1)

            # 学習済の畳み込みニューラルネットワークから分類結果を取得
    result = model.predict_classes(X, verbose=0)
            # 分類結果を表示
    print(janken_class[result[0]])

            # 得られた二値化画像を画面に表示
    cv2.imshow('hand', hand)

            # 'q'を入力でアプリケーション終了
    key = cv2.waitKey(1)
    if key & 0xFF == ord('q'):
        break



cv2.destroyAllWindows()
