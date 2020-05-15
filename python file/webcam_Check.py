import cv2
import numpy as np
from PIL import Image
import matplotlib.pyplot as plt
def before():
    #画像の読み込み
    im = Image.open("./pic.jpg")

    #表示
    im.show()

def after():
    t = 127

    # 入力画像の読み込み
    img = cv2.imread("pic.jpg")

    # グレースケール変換
    gray = cv2.cvtColor(img, cv2.COLOR_RGB2GRAY)

    # 出力画像用の配列生成
    th1 = gray.copy()

    # 方法1（NumPyで実装）
    th1[gray < t] = 0
    th1[gray >= t] = 255

    cv2.imwrite("after.jpg", th1)
    after = Image.open("./after.jpg")
    after.show()


if __name__ == "__main__":
    #key = cv2.waitKey(1) & 0xFF
    #if key == ord('q'):
    # qが押された場合は終了する
        after()
    #elif key == ord('a'):
        before()
