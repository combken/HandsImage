# -*- coding: utf-8 -*-
import cv2
import sys
import numpy as np
import time
import stopWatch
import webcam_getVector as getvc
from sklearn.neural_network import MLPClassifier
from sklearn.externals import joblib
import matplotlib.pyplot as plt
from PIL import Image

start = time.time()
# 学習に用いる縮小画像のサイズ
sw = 16
sh = 12

# 学習結果を保存するファイルの決定
if len(sys.argv)!=2:
    print('使用法: webcam_learn.py 保存ファイル名.pkl')
    sys.exit()
savefile = sys.argv[1]

X = np.empty((0,sw*sh), float)#画像から計算したベクトル
y = np.array([], int)#教師データ

# 画像の読み込み
for hand_class in [0, 1, 2, 3]: # ０がグー　１がパー 2がチー 3がつまみ
    # 画像番号0から999まで対応
    for i in range(1000):
        if hand_class==0: #グー画像
            filename = 'ml-learn/img_gu{0:03d}.png'.format(i)
        elif hand_class==1: #パー画像
            filename = 'ml-learn/img_pa{0:03d}.png'.format(i)
        elif hand_class==2: #チー画像
            filename = 'ml-learn/img_choki{0:03d}.png'.format(i)
        elif hand_class==3: #つまみ画像
            filename = 'ml-learn/img_tsumami{0:03d}.png'.format(i)

        print('{0}を読み込んでいます'.format(filename))
        img = cv2.imread(filename, cv2.IMREAD_GRAYSCALE)
        if img is None:
            break
        # 画像サイズ(lx, ly)と、回転中心座標(cx, cy)の取得
        ly, lx = img.shape[0:2]
        cx, cy = lx/2, ly/2
        # 学習データの格納
        for flip in [0, 1]: # 左右反転なし(0)とあり(1)
            if flip == 1:
                img = cv2.flip(img, 1)
            for angle in [-80, -60, -40, -20, 0, 20, 40, 60, 80]: #角度
                # 回転行列準備
                rot_mat = cv2.getRotationMatrix2D((cx, cy), angle, 1.0)
                # 画像の回転
                img_rot = cv2.warpAffine(img, rot_mat, (lx, ly), flags=cv2.INTER_CUBIC)
                # 回転された画像から、学習用ベクトルの取得
                img_vector = getvc.getImageVector(img_rot,sw,sh)
                # 学習用データの格納
                X = np.append(X, img_vector, axis=0)
                y = np.append(y, hand_class)
# ニューラルネットワークによる画像の学習
clf = MLPClassifier(hidden_layer_sizes=(100, ), max_iter=500, tol=0.0001, random_state=None)

print('学習中…')
clf.fit(X, y)

# 学習結果のファイルへの書き出し
joblib.dump(clf, savefile)
print('学習結果はファイル {0} に保存されました'.format(savefile))
end = time.time()
learn_time = stopWatch.stime(start,end)
print("elapsed_time:{0}".format(learn_time) + "[sec]")
# 損失関数のグラフの軸ラベルを設定
plt.xlabel('time step')
plt.ylabel('loss')

# グラフ縦軸の範囲を0以上と定める
plt.ylim(0, max(clf.loss_curve_))

# 損失関数の時間変化を描画
plt.plot(clf.loss_curve_)

# 描画したグラフを表示
plt.show()
