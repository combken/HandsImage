
"""
ニューラルネットワークのテストを行った
"""
# -*- coding: utf-8 -*-
import cv2
import sys
import time
import stopWatch
import numpy as np
import test_whiteblack as w_b
import webcam_getVector as getvc

from scipy import stats
from sklearn.neural_network import MLPClassifier
from sklearn.externals import joblib

sw = 16
sh = 12 # 学習に用いる縮小画像のサイズ

hmin = 0 # 手の認識用パラメータ（HチャンネルとSチャンネルとを二値化するための条件）
hmax = 30 # 15-40程度にセット
smin = 50
janken_class =  ['グー', 'パー','チー','つまみ','グッド'] #手の形のパターン
print(janken_class)
if len(sys.argv)==2: # 学習済ファイルの確認
    savefile = sys.argv[1]
    try:
        clf = joblib.load(savefile)
    except IOError:
        print('学習済ファイル{0}を開けません'.format(savefile))
        sys.exit()
else:
    print('使用法: python webcam_recognition.py 学習済ファイル.pkl')
    print("学習済みファイルを入力してください")
    sys.exit()

print('認識を開始します')
i = 0
form = 'good'
yes = 0
no = 0
start = time.time()
while i < 100:
    hand = w_b.path_white_black("save_testimages5/%s/img_test%s%03d.png"%(form,form,i),form)# 二値化画像を得る
    cv2.imshow('hand', hand)# 得られた二値化画像を画面に表示
    # 最大の白領域からscikit-learnに入力するためのベクトルを取得
    hand_vector = getvc.getImageVector(hand,sw,sh)#画像を認識しやすいように加工
    result = clf.predict(hand_vector)# 学習済のニューラルネットワークから分類結果を取得
    print(janken_class[result[0]])# 分類結果を表示

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
