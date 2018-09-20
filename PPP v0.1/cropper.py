import numpy as np
import cv2, os, sys, getopt, argparse

pathIMG = ''
pathDIR = ''
pathSAV = ''

drawing = False
cropped = False
crop_selected = False
ix,iy = -1,-1
img_index = 0
rects = []
saved_files = []

def draw(event,x,y,flags,param):
    global ix, iy, drawing, img, backup, cropped, crop_selected

    if cropped == False:

        if event == cv2.EVENT_LBUTTONDOWN:
            drawing = True
            ix,iy = x,y

        elif event == cv2.EVENT_MOUSEMOVE:
            if drawing == True:
                img = backup.copy()
                cv2.imshow(pathIMG,img)
                if abs(ix - x) < abs(iy -y):
                    cv2.rectangle(img,(ix,iy),(x,y),(0,255,0),2)
                else:
                    cv2.rectangle(img,(ix,iy),(x,y),(0,255,0),2)

        elif event == cv2.EVENT_LBUTTONUP:
            drawing = False
            rects.append(((ix,iy),(x,y)))
            while 1:
                for i in rects:
                    cv2.rectangle(img,(i[0][0], i[0][1]),(i[1][0], i[1][1]),(0,255,0),2)
                cv2.imshow(pathIMG, img)
                k = cv2.waitKey(1) & 0xFF
                if k == ord('c'):
                    crop_selected = True
                    for i in rects:
                        crop(i[0][0], i[0][1], i[1][0], i[1][1])
                    if len(saved_files) >= 1:
                        for i in os.listdir(os.getcwd()+'\cropper_test\cropped\\'):
                            os.remove(os.getcwd()+'\cropper_test\cropped\\'+i)
                        for i in saved_files:
                            save(i[0], i[1])
                    quit()                    
                elif k == 27:
                    cv2.destroyAllWindows()
                    print('cancel code')
                    quit()

def getArgvs(argv):
    global pathIMG, pathDIR, pathSAV
    ap = argparse.ArgumentParser()
    ap.add_argument("-i", "--image", required=True, help="Path to the image")
    ap.add_argument("-f", "--folder", required=True, help='Path to the folder')
    ap.add_argument('-s', '--save', required=True, help='Path to the save')
    args = vars(ap.parse_args())
    pathIMG = args['image']
    pathDIR = args['folder']
    pathSAV = args['save']

def crop(ix,iy,x,y):
    global img, backup, cropped, crop_selected

    img = backup.copy()
    cropped = True

    if abs(ix - x) < abs(iy -y):
        img = img[iy:y, ix:x]
        cv2.imshow(pathIMG, img)
        k = cv2.waitKey(0) & 0xFF
        if k == ord('s'):
            img_index = rects.index(((ix, iy), (x, y)))
            saved_files.append((img,img_index))
        elif k == 27:
            print('cancel code')
            quit()
        cv2.destroyAllWindows()

    else:
        img = img[iy:y, ix:x]
        cv2.imshow(pathIMG, img)
        k = cv2.waitKey(0) & 0xFF
        if k == ord('s'):
            img_index = rects.index(((ix, iy), (x, y)))
            saved_files.append((img, img_index))
        elif k == 27:
            print('cancel code')
            quit()
        cv2.destroyAllWindows()

# def crop_multi

def save(crop_fin, img_index):
    new_img = pathSAV + "img" + str(img_index) + ".jpg"
    cv2.imwrite(new_img,crop_fin)

def execute():
    global img

    cv2.namedWindow(pathIMG)
    cv2.setMouseCallback(pathIMG,draw)

    while (1):
        cv2.imshow(pathIMG,img)
        k = cv2.waitKey(1) & 0xFF
        if (k == 27):
            break
        elif (k == ord('s')):
            save(img)
            break
    cv2.destroyAllWindows()

def getIMG(path):
    global img, backup, pathIMG
    path += pathIMG
    pathIMG = path
    img = cv2.imread(pathIMG,-1)
    backup = img.copy()
    scale_percent = 40
    width = int(img.shape[1] * scale_percent / 100)
    height = int(img.shape[0] * scale_percent / 100)
    dim = (width, height)
    resized = cv2.resize(img, dim, interpolation = cv2.INTER_AREA)
    backup = resized.copy()
    img = backup.copy()
    
    execute()

    return 0

def main():
    getArgvs(sys.argv[1:])
    global img, backup, pathIMG

    if pathDIR != '':
        getIMG(pathDIR)
    elif pathIMG != '':
        img = cv2.imread(pathIMG,-1)
        cv2.imshow(pathIMG, img)

        execute()


if __name__ == '__main__':
    main()
    cv2.destroyAllWindows()