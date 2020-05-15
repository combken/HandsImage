
"""
ディープラーニングの学習を行った
"""

# -*- coding: utf-8 -*-
import cv2
import keras
import webcam_getVector as getvc
from keras.datasets import mnist
from keras.models import Sequential
from keras.layers import Dense, Dropout, Flatten, Conv2D, MaxPooling2D
from keras import backend as K
import numpy as np
from scipy import stats
import matplotlib.pyplot as plt
import sys
import time
import stopWatch

start  = time.time()
# keras用のパラメータ
batch_size = 128
epochs = 500
n_classes = 5  #データ種類数に応じて変更する

# 手の二値化画像のサイズ 縦(row)と横(col)
img_rows, img_cols = 12, 16

# 学習に用いる縮小画像のサイズ
sw = img_cols
sh = img_rows

# 学習結果を保存するファイルの決定
if len(sys.argv)==1:
    print('使用法: python ml-10-07-hand-cnn-learn.py 保存ファイル名.h5')
    sys.exit()
savefile = sys.argv[1]
# X:画像から計算したベクトル、y:教師データ
X = np.empty((0,sw*sh), float)
y = np.array([], int)

# グー、チョキ、パー画像の読み込み
for hand_class in [0, 1, 2, 3, 4]: # 0:グー、1:パー、2:チー、3:つまみ、4:グッド
    # 画像番号0から999まで対応
    for i in range(1000):
        if hand_class==0: #グー画像
            filename = 'ml-learn/img_gu{0:03d}.png'.format(i)
        elif hand_class==1: #パー画像
            filename = 'ml-learn/img_pa{0:03d}.png'.format(i)
        elif hand_class==2:#チー画像
            filename = 'ml-learn/img_choki{0:03d}.png'.format(i)
        elif hand_class==3:#つまみ画像
            filename = 'ml-learn/img_tsumami{0:03d}.png'.format(i)
        elif hand_class==4:#グッド画像
            filename = 'ml-learn/img_good{0:03d}.png'.format(i)
        img = cv2.imread(filename, cv2.IMREAD_GRAYSCALE)
        if img is None: #データがなくなったら画像の読み込みを終了させる
            break
        print('{0}を読み込んでいます'.format(filename))
        # 画像サイズ(lx, ly)と、回転中心座標(cx, cy)の取得
        ly, lx = img.shape[0:2]
        cx, cy = lx/2, ly/2
        # 学習データの格納
        for flip in [0, 1]: # 左右反転なし(0)とあり(1)
            if flip == 1:
                img = cv2.flip(img, 1)
            for angle in [-80, -70, -60, -50, -40, -30, -20, -10, 0, 10, 20, 30, 40, 50, 60, 70, 80]: #角度
                # 回転行列準備
                rot_mat = cv2.getRotationMatrix2D((cx, cy), angle, 1.0)
                # 画像の回転
                img_rot = cv2.warpAffine(img, rot_mat, (lx, ly), flags=cv2.INTER_CUBIC)
                # 回転された画像から、学習用ベクトルの取得
                img_vector = getvc.getImageVector(img_rot,sw,sh)
                # 学習用データの格納
                X = np.append(X, img_vector, axis=0)
                y = np.append(y, hand_class)

# データXをCNN用の形式に変換
if K.image_data_format() == 'channels_first':
    X = X.reshape(X.shape[0], 1, img_rows, img_cols)
    input_shape = (1, img_rows, img_cols)
else:
    X = X.reshape(X.shape[0], img_rows, img_cols, 1)
    input_shape = (img_rows, img_cols, 1)
# ターゲットyをkeras用の形式に変換
y_keras = keras.utils.to_categorical(y, n_classes)

# 畳み込みニューラルネットワークを定義
model = Sequential()
model.add(Conv2D(filters=32, kernel_size=(3, 3), activation='relu', input_shape=input_shape))
model.add(Conv2D(filters=64, kernel_size=(3, 3), activation='relu'))
model.add(MaxPooling2D(pool_size=(2, 2)))
model.add(Dropout(0.25))
model.add(Flatten())
model.add(Dense(units=128, activation='relu'))
model.add(Dropout(0.5))
model.add(Dense(units=n_classes, activation='softmax'))

# モデルのコンパイル
model.compile(loss=keras.losses.categorical_crossentropy, optimizer=keras.optimizers.Adadelta(), metrics=['accuracy'])

# モデルの学習
history = model.fit(X, y_keras, batch_size=batch_size, epochs=epochs, validation_split=0., verbose=2)

# 結果の表示
result = model.predict_classes(X, verbose=0)

# データ数をtotalに格納
total = len(X)
# ターゲット（正解）と予測が一致した数をsuccessに格納
success = sum(result==y)

# 正解率をパーセント表示
print('正解率')
print(100.0*success/total)

#時間計測
end = time.time()
delearn_time = stopWatch.stime(start,end)
print("elapsed_time:{0}".format(delearn_time) + "[sec]")#時間を表示
