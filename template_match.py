import cv2
import pathlib
import numpy as np


def get_template(img_path):
    img = cv2.imread(img_path, cv2.IMREAD_GRAYSCALE)

    ret, thresh = cv2.threshold(img, 127, 255, cv2.THRESH_BINARY_INV)

    contours, _ = cv2.findContours(
        thresh,
        cv2.RETR_TREE,
        cv2.CHAIN_APPROX_SIMPLE
    )

    contour = max(contours, key=cv2.contourArea)

    x, y, w, h = cv2.boundingRect(contour)
    print(x, y)
    template = img[y:y + h, x:x + w]
    cv2.imshow("template", template)
    cv2.waitKey(0)
    cv2.destroyAllWindows()
    return template


def main():
    data_dir = "data/dataset/basic"
    data_dir = pathlib.Path(data_dir)

    template = get_template('data/basic/dataset/emoji_0.jpg')

    for img_path in list(data_dir.glob('*.jpg')):
        img = cv2.imread(img_path) 
        res = cv2.matchTemplate(img, template, cv2.TM_CCOEFF_NORMED)
        loc = np.where(res > 0.8)
        print(f"found: {img_path} [{loc}]")


if __name__ == "__main__":
    main()