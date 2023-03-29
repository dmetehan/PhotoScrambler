import cv2
import imageio


def motion(img, img_lab, x, y):
    if 0 < y < img.shape[0] - 1:
        if img_lab[y][x][0] < img_lab[y-1][x][0]:
            img[y-1][x], img[y][x] = img[y][x].copy(), img[y-1][x].copy()
            return True
        elif img_lab[y+1][x][0] < img_lab[y][x][0]:
            img[y+1][x], img[y][x] = img[y][x].copy(), img[y+1][x].copy()
            return True
    elif y == 0:
        if img_lab[y+1][x][0] < img_lab[y][x][0]:
            img[y+1][x], img[y][x] = img[y][x].copy(), img[y+1][x].copy()
            return True
    elif y == img.shape[1] - 1:
        if img_lab[y][x][0] < img_lab[y-1][x][0]:
            img[y-1][x], img[y][x] = img[y][x].copy(), img[y-1][x].copy()
            return True
    return False


path = "test.jpg"
img = cv2.imread(path)

scale = 0.25
width, height = int(round(img.shape[1] * scale)), int(round(img.shape[0] * scale))
img = cv2.resize(img, (width, height), interpolation=cv2.INTER_AREA)
img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
img_lst = [img.copy()] * 60
i = 0
while len(img_lst) < 600:
    img_lab = cv2.cvtColor(img, cv2.COLOR_RGB2LAB)
    for x in range(img.shape[1]):
        for y in range(i, img.shape[0], 3):
            motion(img, img_lab, x, y)
    i = (i + 1) % 3
    img_lst.append(img.copy())
imageio.mimsave('test.gif', img_lst, fps=30)
imageio.mimsave('test_reverse.gif', img_lst[::-1], fps=30)
