
"""
ディープラーニングのテストを行った
"""



# -*- coding: utf-8 -*-
import keras
import stopWatch
import time
from keras import backend as K
import test_whiteblack as w_b
import webcam_getVector as getvc
import webcam_screenshot as getss
import cv2
import numpy as np
from scipy import stats
import sys


# 手の二値化画像のサイズ 縦(row)と横(col)
img_rows, img_cols = 12, 16

# 学習に用いる縮小画像のサイズ
sw = img_cols
sh = img_rows

# 手の認識用パラメータ（HチャンネルとSチャンネルとを二値化するための条件）
hmin = 0
hmax = 30 # 15-40程度にセット
smin = 50

janken_class =  ['グー', 'パー','チー','つまみ','グッド']

# 学習済ファイルの確認
if len(sys.argv)==1:
    print('使用法: webcam_deep_recognition.py 学習済ファイル名.h5')
    sys.exit()
savefile = sys.argv[1]

# 学習済ファイルを読み込んでmodelを作成
model = keras.models.load_model(savefile)

print('認識を開始します')

i = 0
form = 'good'
yes = 0
no = 0
start = time.time()
# 最大の白領域からkerasに入力するためのベクトルを取得
while i < 100:
    hand = w_b.path_white_black("save_testimages5/%s/img_test%s%03d.png"%(form,form,i),form)
    cv2.imshow('hand', hand)
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
    if janken_class[result[0]] == "グッド" :
        yes += 1
    else:
        no += 1
    i += 1


end = time.time()
test_time = stopWatch.stime(start,end)
print("{}[sec]".format(test_time))
print ('end')
print("テスト回数:{}".format(i))
print("正解数:{}  間違い数:{}".format(yes,no))
print("正答率:{}%".format(yes/i*100))

cv2.destroyAllWindows()
