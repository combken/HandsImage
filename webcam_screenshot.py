

def getScreenshot():
    import numpy as np
    import cv2
    from PIL import Image

    cap = cv2.VideoCapture(0)
    while(True):
        ret, frame = cap.read() #フレームをキャプチャする
        cv2.imshow('frame',frame) #画面に表示する
        key = cv2.waitKey(1) & 0xFF#キーボード入力待ち
        if key == ord('q'):#qが押された場合は終了する
            break
        if key == ord('s'):#sが押された場合は保存する
            path = "save_images/pic.jpg"
            cv2.imwrite(path,frame)
            print("スクリーンショットが保存されました")
            break
    cap.release()
    cv2.destroyAllWindows()
