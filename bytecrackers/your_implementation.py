import cv2
import numpy as np
from time import sleep



def find_multiple_instances(base_kp, test_kp, good_matches, min_matches=3, max_instances=10):
    instances = []
    working_matches = good_matches.copy()
    
    for i in range(max_instances):
        if len(working_matches) < min_matches:
            break
            
        src_pts = np.float32([base_kp[m.queryIdx].pt for m in working_matches]).reshape(-1, 2)
        dst_pts = np.float32([test_kp[m.trainIdx].pt for m in working_matches]).reshape(-1, 2)
        
        M, mask = cv2.findHomography(src_pts, dst_pts, cv2.RANSAC, 3.0)
        if M is None:
            break
            
        inliers = [working_matches[i] for i in range(len(working_matches)) if mask[i][0] > 0]
        
        if len(inliers) >= min_matches:
            instances.append((M, inliers))
            
            inlier_indices = set([m.trainIdx for m in inliers])
            working_matches = [m for m in working_matches if m.trainIdx not in inlier_indices]
    
    return instances

def preprocess_happy_emoji(img):
    clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8,8))
    enhanced = clahe.apply(img)
    kernel = np.array([[-1,-1,-1], [-1,9,-1], [-1,-1,-1]])
    sharpened = cv2.filter2D(enhanced, -1, kernel)
    
    return sharpened

def loop_main(X):
    try:
        emoji_names = ['angry', 'crying', 'sad', 'surprised', 'happy']
        base_emojis = [cv2.imread(f'data/emojis/{emoji}.jpg', cv2.IMREAD_GRAYSCALE) 
                    for emoji in emoji_names]
        
        test_image = cv2.imread(f'data/validation/dataset/emoji_{X}.jpg', cv2.IMREAD_GRAYSCALE)
        if test_image is None:
            print(f"FATAL ERROR: NO SUCH IMAGE")
            return

        sift = cv2.SIFT_create()

        base_keypoints = []
        base_descriptors = []
        for i, emoji in enumerate(base_emojis):
            if emoji_names[i] == 'happy':
                emoji = preprocess_happy_emoji(emoji)
            kp, des = sift.detectAndCompute(emoji, None)
            base_keypoints.append(kp)
            base_descriptors.append(des)

        test_keypoints, test_descriptors = sift.detectAndCompute(test_image, None)
        if test_descriptors is None:
            return

        FLANN_INDEX_KDTREE = 1
        index_params = dict(algorithm=FLANN_INDEX_KDTREE, trees=70)
        search_params = dict(checks=80)
        flann = cv2.FlannBasedMatcher(index_params, search_params)
        
        print(f"Picture: emoji_{X}.jpg")

        for i, des in enumerate(base_descriptors):
            if des is None or len(des) < 2: 
                continue
                
            try:
                matches = flann.knnMatch(des, test_descriptors, k=2)
                
                good_matches = []
                for match_pair in matches:
                    if len(match_pair) == 2:
                        m, n = match_pair
                        if m.distance < 0.7 * n.distance:
                            good_matches.append(m)
                
                if len(good_matches) >= 4:
                    instances = find_multiple_instances(
                        base_keypoints[i], test_keypoints, good_matches, min_matches=4
                    )
                    
                    for idx, (M, inliers) in enumerate(instances):
                        h, w = base_emojis[i].shape
                        pts = np.float32([[0, 0], [0, h-1], [w-1, h-1], [w-1, 0]]).reshape(-1, 1, 2)
                        dst = cv2.perspectiveTransform(pts, M)

                        x = int(min(dst[0][0][0], dst[1][0][0], dst[2][0][0], dst[3][0][0]))
                        y = int(min(dst[0][0][1], dst[1][0][1], dst[2][0][1], dst[3][0][1]))
                        
                        if -1000 < x < 1000 and -1000 < y < 1000:
                            print(f"Emoji: {emoji_names[i]} Coordinates: ({x}, {y})")
                    
            except cv2.error:
                continue
            except Exception as e:
                continue

    except Exception as e:
        print(f"Picture: emoji_{X}.jpg\nFAILURE: {str(e)}")

def implementation_main():
    for x in range(0, 281):
        if x == 111:
            continue
        else:
            loop_main(x)
        sleep(0.1)

if __name__ == "__main__":
    implementation_main()