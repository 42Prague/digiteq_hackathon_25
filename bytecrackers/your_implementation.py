import cv2
import numpy as np

def loop_main(X):
    try:
        emoji_names = ['angry', 'crying', 'happy', 'sad', 'surprised']
        base_emojis = [cv2.imread(f'data/emojis/{emoji}.jpg', cv2.IMREAD_GRAYSCALE) 
                    for emoji in emoji_names]

        test_image = cv2.imread(f'data/train/dataset/emoji_{X}.jpg', cv2.IMREAD_GRAYSCALE)
        if test_image is None:
            print(f"FATAL ERROR: NO SUCH IMAGE")
            return

        sift = cv2.SIFT_create()

        base_keypoints = []
        base_descriptors = []
        for emoji in base_emojis:
            kp, des = sift.detectAndCompute(emoji, None)
            base_keypoints.append(kp)
            base_descriptors.append(des)

        test_keypoints, test_descriptors = sift.detectAndCompute(test_image, None)
        if test_descriptors is None:
            # print(f"Picture: emoji_{X}.jpg\nFAILURE: No features found in image")
            return

        bf = cv2.BFMatcher(cv2.NORM_L2, crossCheck=True)
        print(f"Picture: emoji_{X}.jpg")

        for i, des in enumerate(base_descriptors):
            try:
                matches = bf.match(des, test_descriptors)
                matches = sorted(matches, key=lambda x: x.distance)

                if len(matches) > 10:
                    src_pts = np.float32([base_keypoints[i][m.queryIdx].pt for m in matches]).reshape(-1, 2)
                    dst_pts = np.float32([test_keypoints[m.trainIdx].pt for m in matches]).reshape(-1, 2)

                    M, mask = cv2.findHomography(src_pts, dst_pts, cv2.RANSAC, 5.0)
                    if M is None:
                        continue

                    h, w = base_emojis[i].shape
                    pts = np.float32([[0, 0], [0, h-1], [w-1, h-1], [w-1, 0]]).reshape(-1, 1, 2)
                    dst = cv2.perspectiveTransform(pts, M)

                    x = int(min(dst[0][0][0], dst[1][0][0], dst[2][0][0], dst[3][0][0]))
                    y = int(min(dst[0][0][1], dst[1][0][1], dst[2][0][1], dst[3][0][1]))
                    
                    if -1000 < x < 1000 and -1000 < y < 1000:  # Basic sanity check
                        print(f"Emoji: {emoji_names[i]} Coordinates: ({x}, {y})")
                    
            except cv2.error:
                continue
            except Exception as e:
                continue
                # print(f"FAILURE: Error processing {emoji_names[i]} - {str(e)}")

    except Exception as e:
        print(f"Picture: emoji_{X}.jpg\nFAILURE: {str(e)}")

def implementation_main():
    for x in range(0, 1121):  # Changed to start from 0 and include 1120
        loop_main(x)

if __name__ == "__main__":
    implementation_main()
