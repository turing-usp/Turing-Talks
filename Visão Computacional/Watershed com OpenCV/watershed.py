import cv2 as cv
import numpy as np
import matplotlib.pyplot as plt

#leitura da imagem
img_name = "images/tomatos.jpg"
img = cv.imread(img_name)

#converter imagem para preto e branco
gray = cv.cvtColor(img, cv.COLOR_BGR2GRAY)

#thresholding da imagem
_, thresh = cv.threshold(gray, 0, 255, cv.THRESH_BINARY+cv.THRESH_OTSU)

'''
#Código para gerar comparacao de transformacoes morfologicas

kernel = np.ones((3,3), np.uint8)

tomates = read_image("images/tomatos.jpg")

dilated = cv.dilate(tomates, kernel, iterations = 3)
eroded = cv.erode(tomates, kernel, iterations = 3)
opening = cv.morphologyEx(tomates, cv.MORPH_OPEN, kernel, iterations = 5)


fig, axs = plt.subplots(2, 2)


axs[0][0].imshow(tomates, cmap="gray")
axs[0][0].set_title("Original")


axs[0][1].imshow(dilated, cmap="gray")
axs[0][1].set_title("Dilated")

axs[1][0].imshow(eroded, cmap="gray")
axs[1][0].set_title("Eroded")

axs[1][1].imshow(opening, cmap="gray")
axs[1][1].set_title("Opening")

plt.savefig("comparison2.jpg", transparent=True)
'''

#opening: erosion seguida de dilation. Retira ruido da imagem
kernel = np.ones((3,3), np.uint8)
opening = cv.morphologyEx(thresh, cv.MORPH_OPEN, kernel, iterations=10)

#background
sure_bg = cv.dilate(opening, kernel, iterations=10)

#foreground
#distancia do foreground para o background de cada pixel
dist = cv.distanceTransform(opening, cv.DIST_L2, 5)


#threshold nos diz o que temos certeza que esta no foreground
_, sure_fg = cv.threshold(dist, 0, 255, cv.THRESH_BINARY)
sure_fg = np.uint8(sure_fg)

#pixels desconhecidos
unknown = cv.subtract(sure_bg, sure_fg)

#cricacao dos marcadores
_, markers = cv.connectedComponents(sure_fg)

markers = markers + 1

markers[unknown==255] = 0

markers = cv.watershed(img, markers)
img[markers == -1] = [255,0,0]

file_name = "watershed.jpg"
cv.imwrite(file_name, img)