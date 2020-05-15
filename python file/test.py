import cv2
from PIL import Image, ImageOps
from PIL import ImageFont, ImageDraw

for i in range(1000):
    filename = 'choki_image/img_choki001png'
    img = cv2.imread(filename,cv2.IMREAD_GRAYSCALE)
    img_copy = img.copy()
    img_mirror = cv2.flip(img_copy, 1)
    cv2.imwrite('left_choki_image/img_chokileft001.png',img_mirror)

    if img is None:
        break
