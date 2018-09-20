'''
Using a modified code from AVatch's Github Repository
https://gist.github.com/AVatch/843a9e5ed9cd1214676c849c7ff6736d#file-sift-py
'''

import cv2, os, sys
import numpy as np
import matplotlib.pyplot as plt
import argparse

# set the "canvas" as the imported solved image of choice
canvas = cv2.imread('grab.jpg')
# set the piece as the imported image of choice
piece = cv2.imread('crop_img.jpg')


def main(c, p):  # will eventually be executed multiple times when solving entire puzzles

    def identify_contour(img, threshold_low=150, threshold_high=255):  # pre-processing
        """
            this is the process i used for identifying the contours about the images
        """
        img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)  # adjust to greyscale to eliminate weak keypoints
        ret, thresh = cv2.threshold(img, threshold_low, threshold_high, 0)
        image, contours, hierarchy = cv2.findContours(thresh, cv2.RETR_TREE, 1)
        # print(contours)  # uncomment this to see the list of contours provided to the sort
        contour_sorted = np.argsort(list(map(cv2.contourArea, contours)))
        print(contour_sorted)
        return contours, contour_sorted[-2]

    def get_bounding_rect(contour):
        """
            the helpful bit that isolates the useful parts of the "piece" image and reduces it's size
        """
        x, y, w, h = cv2.boundingRect(contour)
        return x, y, w, h

    # using the pre-processing function on both the piece and the canvas
    contours, contour_index = identify_contour(piece.copy())

    # makes the bounding area to reduce size and increase efficiency
    x, y, w, h = get_bounding_rect(contours[contour_index])
    cropped_piece = piece.copy()[y:y+h, x:x+w]

    # creates the SIFT detector instance
    sift = cv2.xfeatures2d.SIFT_create()

    img1 = cropped_piece.copy()  # the piece, also queryImage
    img2 = canvas.copy()  # the canvas, also trainImage

    # running SIFT on the two images
    kp1, des1 = sift.detectAndCompute(img1, None)
    kp2, des2 = sift.detectAndCompute(img2, None)


    """
        alright so this is how it knows what to match at what order, so it will take the params
        and match them to make the process quicker
    """
    FLANN_INDEX_KDTREE = 0
    index_params = dict(algorithm=FLANN_INDEX_KDTREE, trees=5)
    search_params = dict(checks=50)

    flann = cv2.FlannBasedMatcher(index_params, search_params)

    matches = flann.knnMatch(des1, des2, k=2)

    # Lowe's ratio test allows us to store good matches, so thats what we do
    good = []  # good boyes
    for m, n in matches:
        if m.distance < 0.7*n.distance:  # making sure that it is in the right range
            good.append(m)

    MIN_MATCH_COUNT = 10  # this is just a number, can be changed, but you can get 7 wrong strong matches,
    # but 10 is highly unlikely, it's also a nice number

    if len(good) > MIN_MATCH_COUNT:
        """
            this is how the green lines are drawn, it takes the positions of each match and feeds it into OpenCV powered functions
        """
        src_pts = np.float32([kp1[m.queryIdx].pt for m in good]).reshape(-1, 1, 2)
        dst_pts = np.float32([kp2[m.trainIdx].pt for m in good]).reshape(-1, 1, 2)

        M, mask = cv2.findHomography(src_pts, dst_pts, cv2.RANSAC,5.0)
        matchesMask = mask.ravel().tolist()

        d, h, w = img1.shape[::-1]
        pts = np.float32([[0, 0], [0, h-1], [w-1, h-1], [w-1, 0]]).reshape(-1, 1, 2)
        dst = cv2.perspectiveTransform(pts, M)

        img2 = cv2.polylines(img2, [np.int32(dst)], True, 255, 3, cv2.LINE_AA)
        print("Good match! - %d/%d" % (len(good), MIN_MATCH_COUNT))

    else:
        print("Not enough matches are found - %d/%d" % (len(good), MIN_MATCH_COUNT))
        matchesMask = None

    draw_params = dict(matchColor=(0, 255, 0),  # draw good matches in green color
                       singlePointColor=None,
                       matchesMask=matchesMask,  # draw only inliers (within the blue)
                       flags=cv2.DRAW_MATCHES_FLAGS_DRAW_RICH_KEYPOINTS)

    img3 = cv2.drawMatches(img1, kp1, img2, kp2, good, None, **draw_params)

    """
        you can actually delete the "solution.jpg" file, and it will make a new one,
        so don't worry if you do
    """
    cv2.imwrite('solution.jpg', img3)  
    solution = cv2.imread('solution.jpg')
    width, height, channels = solution.shape
    solution = cv2.resize(solution, (int(height/2), int(width/2)))
    cv2.imshow("Result", solution)

    cv2.waitKey(0)
    cv2.destroyAllWindows()


if __name__ == "__main__":
    """
        this will open the piece and the canvas to show you the images you chose,
        they stay open for 3 seconds then close, change the time in the waitKey() function
    """

    #--------------------------------------------------------------------------------------------|

    # This is where you can change what the grabber uses, it will save the file as img01.jpg
    # so when you go to call this, sift-basic.py, you will call it on img01.jpg if you want to
    # use it, just comment these lines out if you don't want to use it :)
    
    os.system("python cropper.py --image test.jpg -f cropp`er_test\\raw\ --save cropper_test\cropped\\")

    #--------------------------------------------------------------------------------------------|

    ap = argparse.ArgumentParser()
    ap.add_argument("-c", "--canvas", required=True, help="Path to the canvas")
    ap.add_argument("-p", "--piece", required=True, help="Path to the piece")
    args = vars(ap.parse_args())
    canvas = cv2.imread(args["canvas"])
    piece = cv2.imread(args["piece"])
    main(canvas, piece)
