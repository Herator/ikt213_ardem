import numpy as np
import cv2

def sobel_edge_detection(image):
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    blur = cv2.GaussianBlur(gray, (3, 3), 0)

    sobel = cv2.Sobel(src=blur, ddepth=cv2.CV_64F, dx=1, dy=1, ksize=1)
    sobel = cv2.convertScaleAbs(sobel)
    return sobel

def canny_edge_detection(image, threshold_1, threshold_2):
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    blur = cv2.GaussianBlur(gray, (3, 3), 0)

    canny = cv2.Canny(blur, threshold_1, threshold_2)
    return canny

def template_match(image, template):
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    template_gray = cv2.cvtColor(template, cv2.COLOR_BGR2GRAY)
    w, h = template_gray.shape[::-1]

    res = cv2.matchTemplate(gray, template_gray, cv2.TM_CCOEFF_NORMED)
    threshold = 0.9
    loc = np.where(res >= threshold)
    for pt in zip(*loc[::-1]):
        cv2.rectangle(image, pt, (pt[0] + w, pt[1] + h), (0, 0, 255), 2)

    return image

def resize(image, scale_factor:int, up_or_down: str):
    for i in range(scale_factor):
        if up_or_down == "up":
            image = cv2.pyrUp(image)
        elif up_or_down == "down":
            image = cv2.pyrDown(image)

    return image


img = cv2.imread("./lambo.png")

# Task 1

sobel = sobel_edge_detection(img)
cv2.imwrite("./solutions/task-1-sobel_egde_detection.png", sobel)

# Task 2
canny = canny_edge_detection(img,50, 50)
cv2.imwrite("./solutions/task-2-canny_egde_detection.png", canny)

# Task 3
shapes = cv2.imread("./shapes.png")
shapes_template = cv2.imread("./shapes_template.jpg")
match = template_match(shapes, shapes_template)
cv2.imwrite("./solutions/task-3-template_match.png", match)

# Task 4
resized = resize(img, 2, "up")
cv2.imwrite("./solutions/task-4_resized.png", resized)