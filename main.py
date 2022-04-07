"""Image Stitching"""

__author__ = "Alihan KARATATAR"
__version__ = "3.8.0 and 3.7.0 (for SURF)"

"*****************************************"

# Importing libraries
import os
import cv2
import numpy as np


def feature_matching(img1, img2):
    # FOR USING SURF SIMPLY CHANGE SIFT_create to SURF_create
    # Important!! Since SURF is patented for python 3.8 and above you should use python 3.7 or lower version

    sift = cv2.SIFT_create()
    # Key point detection
    key_point1, des1 = sift.detectAndCompute(img1, None)
    key_point2, des2 = sift.detectAndCompute(img2, None)

    # I use Flann Index. It gives more accurate result for SIFT than KNN
    FLANN_INDEX_KDTREE = 2
    index_params = dict(algorithm=FLANN_INDEX_KDTREE, trees=5)
    search_params = dict(checks=50)
    flann = cv2.FlannBasedMatcher(index_params, search_params)
    matches2to1 = flann.knnMatch(des2, des1, k=2)

    matchesMask_ratio = [[0, 0] for i in range(len(matches2to1))]
    match_dict = {}
    for i, (m, n) in enumerate(matches2to1):
        if m.distance < 0.7 * n.distance:
            matchesMask_ratio[i] = [1, 0]
            match_dict[m.trainIdx] = m.queryIdx

    arr = []
    matches = flann.knnMatch(des1, des2, k=2)
    matchesMask_ratio_recip = [[0, 0] for j in range(len(matches))]


    # Appending array with good matches after that it will be returned as source and destination points
    for i, (m, n) in enumerate(matches):
        if m.distance < 0.7 * n.distance:
            if m.queryIdx in match_dict and match_dict[m.queryIdx] == m.trainIdx:
                arr.append(m)
                matchesMask_ratio_recip[i] = [1, 0]

    # FOR USING ORB, COMMENT OUT FOLLOWING PART AND COMMENT IN ABOVE PART


    # orb = cv2.ORB_create()
    # key_point1, des1 = orb.detectAndCompute(img1, None)
    # key_point2, des2 = orb.detectAndCompute(img2, None)
    # bf = cv2.BFMatcher()
    # matches = bf.knnMatch(des1, des2, k=2)
    #
    # good_matches = []
    # for m, n in matches:
    #    if m.distance < 0.8 * n.distance:
    #        good_matches.append(m)
    #

    # src_pts = np.float32([key_point1[m.queryIdx].pt for m in good_matches]) \
    #    .reshape(-1, 1, 2)
    # dst_pts = np.float32([key_point2[m.trainIdx].pt for m in good_matches]) \
    #    .reshape(-1, 1, 2)

    # return src_pts, dst_pts
    return [key_point1[m.queryIdx].pt for m in arr], [key_point2[m.trainIdx].pt for m in arr]

# Computing homography matrix M by using RANSAC
def get_transform(src, dst):
    first_points, second_points = feature_matching(src, dst)

    src_pts = np.float32(first_points).reshape(-1, 1, 2)
    dst_pts = np.float32(second_points).reshape(-1, 1, 2)

    # Checking for whether source points lenght above 4 or not.
    if len(src_pts) > 4:
        M, mask = cv2.findHomography(src_pts, dst_pts, cv2.RANSAC)
    else:
    # If not it initalize Homograpy matrix as follow
        M, mask = cv2.findHomography(src_pts, dst_pts, cv2.RANSAC)
        M = np.array([[0, 0, 0], [0, 0, 0], [0, 0, 0]])

    return M, first_points, second_points, mask

# Here we use laplacian pyramids. We use them inside of our blending algorithm.
# This method gives our stitched image fish-eye lens seen.
# It is better for panoramic images
def laplacian_pyramids(img1, img2, mask, levels):
    gaussian_input1 = img1.copy()
    gaussian_input2 = img2.copy()
    gaussian_input3 = mask.copy()
    gaussian_pyramid1 = [gaussian_input1]
    gaussian_pyramid2 = [gaussian_input2]
    gaussian_pyramid3 = [gaussian_input3]

    for i in range(levels):
        # Scaling Down for laplacian image pyramid
        gaussian_input1 = cv2.pyrDown(gaussian_input1)
        gaussian_input2 = cv2.pyrDown(gaussian_input2)
        gaussian_input3 = cv2.pyrDown(gaussian_input3)
        gaussian_pyramid1.append(np.float32(gaussian_input1))
        gaussian_pyramid2.append(np.float32(gaussian_input2))
        gaussian_pyramid3.append(np.float32(gaussian_input3))

    laplacian1 = [gaussian_pyramid1[levels - 1]]
    laplacian2 = [gaussian_pyramid2[levels - 1]]
    laplacian3 = [gaussian_pyramid3[levels - 1]]

    for i in range(levels - 1, 0, -1):
        # Scaling up for laplacian image pyramid
        L1 = np.subtract(gaussian_pyramid1[i - 1], cv2.pyrUp(gaussian_pyramid1[i]))
        L2 = np.subtract(gaussian_pyramid2[i - 1], cv2.pyrUp(gaussian_pyramid2[i]))
        laplacian1.append(L1)
        laplacian2.append(L2)
        laplacian3.append(gaussian_pyramid3[i - 1])

    arr2 = []
    for l1, l2, gm in zip(laplacian1, laplacian2, laplacian3):
        ls = l1 * gm + l2 * (1.0 - gm)
        arr2.append(ls)

    Laplacian_syntheses = arr2[0]

    for i in range(1, levels):
        Laplacian_syntheses = cv2.pyrUp(Laplacian_syntheses)
        Laplacian_syntheses = cv2.add(Laplacian_syntheses, arr2[i])

    return Laplacian_syntheses

# Function that apply perspective warping
def perspective_warping(img1, img2, img3):
    img1 = cv2.copyMakeBorder(img1, 50, 50, 250, 250, cv2.BORDER_CONSTANT)
    (M, first_points, second_points, mask) = get_transform(img3, img1)

    (M1, third_points, forth_points, mask2) = get_transform(img2, img1)

    m = np.ones_like(img3, dtype='float32')
    m1 = np.ones_like(img2, dtype='float32')

    canvas1 = cv2.warpPerspective(img3, M, (img1.shape[1], img1.shape[0]))
    canvas2 = cv2.warpPerspective(img2, M1, (img1.shape[1], img1.shape[0]))
    canvas3 = cv2.warpPerspective(m, M, (img1.shape[1], img1.shape[0]))
    canvas4 = cv2.warpPerspective(m1, M1, (img1.shape[1], img1.shape[0]))

    first_blending = laplacian_pyramids(canvas1, img1, canvas3, 1)
    final_blending = laplacian_pyramids(canvas2, first_blending, canvas4, 1)
    return final_blending

# Path initalization
path = "./part_1_dataset/cvc01passadis-cyl-pano18"
filename = list()

for file in os.listdir(path):
    filename.append(file)

filename = filename[0:33]
mid_img = filename[len(filename) // 2]
# Take the middle image always. Since we have 33 images, it is 17th image
middle = cv2.imread(path + "/" + mid_img)
# Left side of middle
left_im = filename[len(filename) // 2 - 1]
# Rİght side of middle
right_im = filename[len(filename) // 2 + 1]

# Reading left and rigth as images
left = cv2.imread(path + "/" + left_im).astype("uint8")
right = cv2.imread(path + "/" + right_im).astype("uint8")

# For loop to blend all images in directory
for i in range(1, len(filename) // 2):

    middle2 = perspective_warping(middle, left, right)
    # Save image after every blending operation step
    cv2.imwrite(f'./result/18/generated_panorama{i}.png', middle2)

    left_img = filename[len(filename) // 2 - (i + 1)]
    right_img = filename[len(filename) // 2 + (i + 1)]

    left = cv2.imread(path + "/" + left_img).astype("uint8")
    right = cv2.imread(path + "/" + right_img).astype("uint8")
    middle = middle2.astype("uint8")
