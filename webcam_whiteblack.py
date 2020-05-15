# -*- coding: utf-8

def white_black():
    from PIL import Image
    import cv2
    import numpy as np
    from scipy import stats

    hmin = 0 #二値化画像を作成する際の範囲値の設定
    hmax = 30 # 15-40程度にセット
    smin = 50 #

    while True:
        img_src = cv2.imread("save_images/pic.jpg", 1)#テストの場合はここに画像へのパス
        # 映像データをHSV形式に変換
        hsv_src = cv2.cvtColor(img_src, cv2.COLOR_BGR2HSV)
        # HSV形式からHチャンネルとSチャンネルの画像を得る
        hsv_channels = cv2.split(hsv_src)
        h_channel = hsv_channels[0]
        s_channel = hsv_channels[1]

        # Hチャンネルを平滑化
        h_binary = cv2.GaussianBlur(h_channel, (5,5), 0)

        # Hチャンネルの二値化画像を作成
        # hmin～hmaxの範囲を255（白）に、それ以外を0（黒）に
        ret,h_binary = cv2.threshold(h_binary, hmax, 255, cv2.THRESH_TOZERO_INV)
        ret,h_binary = cv2.threshold(h_binary, hmin, 255, cv2.THRESH_BINARY)
        # Sチャンネルの二値化画像を作成
        # smin～255の範囲を255（白）に、それ以外を0に（黒）に
        ret,s_binary = cv2.threshold(s_channel, smin, 255, cv2.THRESH_BINARY)

        # HチャンネルとSチャンネルの二値化画像のANDをとる
        # HチャンネルとSチャンネルの両方で255（白）の領域のみ白となる
        hs_and = h_binary & s_binary
        path = "save_images/after2.jpg" #２値化後の画像を表示
        cv2.imwrite(path,hs_and)
        # 以下、最も広い白領域のみを残すための計算
        # まず、白領域の塊（クラスター）にラベルを振る
        img_dist, img_label = cv2.distanceTransformWithLabels(255-hs_and,cv2.DIST_L2, 5)
        img_label = np.uint8(img_label) & hs_and
        # ラベル0は黒領域なので除外
        img_label_not_zero = img_label[img_label != 0]
        # 最も多く現れたラベルが最も広い白領域のラベル
        if len(img_label_not_zero) != 0:
            m = stats.mode(img_label_not_zero)[0]
        else:
            m = 0
            # 最も広い白領域のみを残す
        hand = np.uint8(img_label == m)*255


        key = cv2.waitKey(1) & 0xFF
        # 得られた二値化画像を画面に表示

        cv2.imshow("hand", hand)
        cv2.moveWindow('hand', 800, 200) #ウィンドウの位置調整
        cv2.imwrite("save_images/white_black.jpg",hand)#最終的なファイルの作成
        if key == ord('q'):
            break
    cv2.destroyAllWindows()

    return hand
