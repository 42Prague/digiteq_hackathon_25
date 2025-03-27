import cv2
import pathlib
import numpy as np
import argparse


def best_matches(matches):
    """ Checks that distance between matches is over 30 pixels
     to avoid multiple classifications """
    matches = sorted(matches, key=lambda x: x[2], reverse=True)
    distance_threshold = 30

    keep = []
    for match in matches:
        if not keep:
            keep.append(match)
            continue

        x, y = match[0], match[1]
        should_keep = True

        for kept_match in keep:
            kept_x, kept_y = kept_match[0], kept_match[1]
            # pythag distance
            distance = np.sqrt((x - kept_x)**2 + (y - kept_y)**2)
            if distance < distance_threshold:
                should_keep = False
                break
        if should_keep:
            keep.append(match)

    return keep


def template_matcher(img, template, rotate=False):
    """ uses match template to find template inside image.

      uses normalized cross-coefficient method as it seems to work best for
      mostly binary images

      when rotate is used, rotate in region of interest found from original
      match.
      if rotation creates better score than base, use rotated result"""
    base_result = cv2.matchTemplate(img, template, cv2.TM_CCOEFF_NORMED)
    best_score = np.max(base_result)
    best_result = base_result

    if rotate is True:
        min_val, max_val, min_loc, max_loc = cv2.minMaxLoc(base_result)
        match_x, match_y = max_loc

        # only focus on found area
        h, w = template.shape
        roi = img[max(0, match_y-h//2):min(img.shape[0], match_y+h*3//2),
                  max(0, match_x-w//2):min(img.shape[1], match_x+w*3//2)]

        if roi.size > 0:  # Check if ROI is valid
            # Check rotations only on ROI
            for angle in range(-45, 46, 5):
                center = (roi.shape[1]//2, roi.shape[0]//2)
                rotation_matrix = cv2.getRotationMatrix2D(center, angle, 1.0)
                rotated = cv2.warpAffine(roi, rotation_matrix,
                                         (roi.shape[1], roi.shape[0]))

                res = cv2.matchTemplate(rotated, template,
                                        cv2.TM_CCOEFF_NORMED)
                score = np.max(res)

                if score > best_score:
                    best_score = score
                    # Keep original result for coordinates
                    best_result = base_result

    return best_result, best_score


def emoji_templates():
    """ creates templates from given emoji set """
    templates = {
        'angry': cv2.imread(
            "data/emojis/angry/angry.jpg", cv2.IMREAD_GRAYSCALE),

        'crying': cv2.imread(
            "data/emojis/crying/crying.jpg", cv2.IMREAD_GRAYSCALE),

        'happy': cv2.imread(
            "data/emojis/happy/happy.jpg", cv2.IMREAD_GRAYSCALE),

        'sad': cv2.imread(
            "data/emojis/sad/sad.jpg", cv2.IMREAD_GRAYSCALE),

        'surprised': cv2.imread(
            "data/emojis/surprised/surprised.jpg", cv2.IMREAD_GRAYSCALE)
    }
    return templates


def preprocess_image(img_path):
    """ uses adaptive threshold to hopefully clean up image for match
    searching """
    img = cv2.imread(str(img_path), cv2.IMREAD_GRAYSCALE)
    img = cv2.adaptiveThreshold(
            img,
            255,
            cv2.ADAPTIVE_THRESH_GAUSSIAN_C,
            cv2.THRESH_BINARY,
            blockSize=11,
            C=2)
    img = cv2.fastNlMeansDenoising(img,
                                   h=5,  # Filter strength (5-30)
                                   templateWindowSize=7,  # Template patch size
                                   searchWindowSize=21)  # Search window size
    return img


def parser():
    parser = argparse.ArgumentParser(description='Emoji template matching')
    parser.add_argument('labels', help='Path to labels.csv file')
    parser.add_argument('-r', '--rotate', action='store_true',
                        help='Enable rotation check during template matching')
    parser.add_argument('-l', '--limit', action='store_true',
                        help="limits tests to 10 imgs")
    parser.add_argument('-t', '--threshold', type=float, default=0.4, 
                        help="changes threshold value "
                        "(value must be 0-1 float), default=0.4")
    parser.add_argument('-b', '--bonus', action="store_true",
                        help="changes to bonus mode")
    args = parser.parse_args()
    return args


def implementation_main():
    """ loops through each image in directory
     uses template matching to find possible emojis, and removes
     potential duplicates.
      prints image name and best found emotion and it's position on image. """

    args = parser()
    labels_path = pathlib.Path(args.labels)
    data_dir = labels_path.parent / "dataset"
    templates = emoji_templates()
    threshold = args.threshold

    limit = 0
    for img_path in list(data_dir.glob('*.jpg')):
        img = preprocess_image(img_path)

        results = []
        for name, template in templates.items():
            res, _ = template_matcher(img, template, args.rotate)
            locs = np.where(res >= threshold)

            for y, x in zip(*locs):
                results.append((x, y, res[y, x], name))

        matches = best_matches(results)

        print(f"Picture: {img_path.name}")
        for x, y, score, emoji_name in matches:
            print(f"Emoji: {emoji_name} Coordinates: ({x}, {y})")

        limit += 1
        if args.limit and limit > 10:
            break


if __name__ == "__main__":
    implementation_main()


# def find_area_of_interest_and_match(img_path, template_size):
#     """ only works for non-bonus imgs """
#     img = cv2.imread(img_path, cv2.IMREAD_GRAYSCALE)
#     ret, thresh = cv2.threshold(img, 200, 255, cv2.THRESH_BINARY_INV)

#     contours, _ = cv2.findContours(
#         thresh,
#         cv2.RETR_TREE,
#         cv2.CHAIN_APPROX_SIMPLE
#     )

#     contour = max(contours, key=cv2.contourArea)

#     x, y, w, h = cv2.boundingRect(contour)
#     crop = img[y:y + h, x:x + w]
#     new_img = cv2.resize(crop, template_size)

#     cv2.imshow("crop", new_img)
#     cv2.waitKey(0)
#     cv2.destroyAllWindows()

#     return new_img