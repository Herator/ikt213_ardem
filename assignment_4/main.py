import numpy as np
import cv2


def Harris(reference_image):
    out = reference_image.copy()
    gray = cv2.cvtColor(out, cv2.COLOR_BGR2GRAY)
    gray = np.float32(gray)

    dst = cv2.cornerHarris(gray, 2, 3, 0.04)
    dst = cv2.dilate(dst, None)

    out[dst > 0.01 * dst.max()] = [0, 0, 255]
    cv2.imwrite("./solutions/harris.png", out)


def alignment(image_to_align, reference_image, max_features, good_match_precent):
    sift = cv2.SIFT_create()

    kp1, des1 = sift.detectAndCompute(reference_image, None)
    kp2, des2 = sift.detectAndCompute(image_to_align, None)

    index_params = dict(algorithm=1, trees=5)
    search_params = dict(checks=50)
    flann = cv2.FlannBasedMatcher(index_params, search_params)

    matches = flann.knnMatch(des1, des2, 2)

    good = []
    for m, n in matches:
        if m.distance < good_match_precent * n.distance:
            good.append(m)

    if len(good) > max_features:
        src_pts = np.float32([kp1[m.queryIdx].pt for m in good]).reshape(-1, 1, 2)
        dst_pts = np.float32([kp2[m.trainIdx].pt for m in good]).reshape(-1, 1, 2)

        # map image_to_align -> reference_image, so we can warp align onto ref's frame
        M, mask = cv2.findHomography(dst_pts, src_pts, cv2.RANSAC, 5.0)
        matchesMask = mask.ravel().tolist() if M is not None else None

        h, w = reference_image.shape[:2]
        aligned = cv2.warpPerspective(image_to_align, M, (w, h))
        cv2.imwrite("./solutions/aligned.png", aligned)
    else:
        print("Not enough matches are found - {}/{}".format(len(good), max_features))
        matchesMask = None

    draw_params = dict(matchColor=(0, 255, 0),
                        singlePointColor=None,
                        matchesMask=matchesMask,
                        flags=2)
    final = cv2.drawMatches(reference_image, kp1, image_to_align, kp2, good, None, **draw_params)
    cv2.imwrite("./solutions/matches.png", final)


ref = cv2.imread("./reference_img.png")
align = cv2.imread("./align_this.jpg")

# Task 1
Harris(ref)

# Task 2: SIFT
alignment(align, ref, 10, 0.7)