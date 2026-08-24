import numpy as np
import cv2

def padding (image, border_width):
    return cv2.copyMakeBorder(image,border_width,border_width,border_width,border_width,cv2.BORDER_REFLECT)

def crop (image,x_0, x_1, y_0, y_1):
    x_1, y_1 = w - x_1, h - y_1
    return image[y_0:y_1,x_0:x_1]

def resize (image, width, height):
    return cv2.resize(image, (width, height))

def copy (image, emptyPictureArray):
    emptyPictureArray[:, :] = image[:, :]
    return emptyPictureArray

def grayscale (image):
    return cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

def hsv (image):
    return cv2.cvtColor(image, cv2.COLOR_BGR2HSV)

def hue_shifted (image, emptyPictureArray, hue):
    emptyPictureArray[:, :] = image + hue
    return emptyPictureArray

def smoothing (image):
    return cv2.GaussianBlur(image, (15,15), cv2.BORDER_DEFAULT)

def rotation (image, rotation_angle):
    if rotation_angle == 90:
        return cv2.rotate(image, cv2.ROTATE_90_CLOCKWISE)
    elif rotation_angle == 180:
        return cv2.rotate(image, cv2.ROTATE_180)


img = cv2.imread("./solutions/iris.png")
h, w, channels = img.shape[:3]

# Task 1
padded = padding(img,100)
cv2.imwrite("./solutions/task-1-padded.png", padded)

# Task 2
cropped = crop(img,200,130,200,130)
cv2.imwrite("./solutions/task-2-cropped.png", cropped)

# Task 3
resized = resize(img,200,200)
cv2.imwrite("./solutions/task-3-resized.png", resized)

# Task 4
emptyPictureArray = np.zeros((h,w,channels), np.uint8)
copyd = copy(img, emptyPictureArray)
cv2.imwrite("./solutions/task-4-copy.png", copyd)

# Task 5
gray = grayscale(copyd)
cv2.imwrite("./solutions/task-5-gray.png", gray)

# Task 6
hsvImage = hsv(img)
cv2.imwrite("./solutions/task-6-hsv.png", hsvImage)

# Task 7
emptyPictureArray = np.zeros((h,w,channels), np.uint8)
shifted = hue_shifted(img,emptyPictureArray,50)
cv2.imwrite("./solutions/task-7-shifted.png", shifted)

# Task 8
smoothed = smoothing(img)
cv2.imwrite("./solutions/task-8-smoothed.png", smoothed)

# Task 9
rotat_90 = rotation(img,90)
rotate_180 = rotation(img,180)
cv2.imwrite("./solutions/task-9-rotated_90.png", rotat_90)
cv2.imwrite("./solutions/task-9-rotated_180.png", rotate_180)