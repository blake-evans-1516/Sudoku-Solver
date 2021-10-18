from PIL import Image
import pygame
import cv2
import numpy as np
import matplotlib.pyplot as plt
from itertools import product
from pytesseract import *

# Initialise Pygame
pygame.init()
pygame.font.init()

# Defining Values
WIDTH = 550
background_color = (251,247,245)
screen = pygame.display.set_mode((500, 600))
x = 0
y = 0

img = cv2.imread('sudoku.jpg')
img = cv2.GaussianBlur(img,(5,5),0)
gray = cv2.cvtColor(img,cv2.COLOR_BGR2GRAY)
mask = np.zeros((gray.shape),np.uint8)
kernel1 = cv2.getStructuringElement(cv2.MORPH_ELLIPSE,(11,11))

close = cv2.morphologyEx(gray,cv2.MORPH_CLOSE,kernel1)
div = np.float32(gray)/(close)
res = np.uint8(cv2.normalize(div,div,0,255,cv2.NORM_MINMAX))
res2 = cv2.cvtColor(res,cv2.COLOR_GRAY2BGR)

# Finding Sudoku Square and Creating Mask Image
thresh = cv2.adaptiveThreshold(res,255,0,1,19,2)
contour,hier = cv2.findContours(thresh,cv2.RETR_TREE,cv2.CHAIN_APPROX_SIMPLE)

max_area = 0
best_cnt = None
for cnt in contour:
    area = cv2.contourArea(cnt)
    if area > 1000:
        if area > max_area:
            max_area = area
            best_cnt = cnt

cv2.drawContours(mask,[best_cnt],0,255,-1)
cv2.drawContours(mask,[best_cnt],0,0,3)

res= cv2.bitwise_and(res,mask)

# Finding Vertical lines
kernelx = cv2.getStructuringElement(cv2.MORPH_RECT,(2,10))

dx = cv2.Sobel(res,cv2.CV_16S,1,0)
dx = cv2.convertScaleAbs(dx)
cv2.normalize(dx,dx,0,255,cv2.NORM_MINMAX)
ret,close = cv2.threshold(dx,0,255,cv2.THRESH_BINARY+cv2.THRESH_OTSU)
close = cv2.morphologyEx(close,cv2.MORPH_DILATE,kernelx,iterations = 1)

contour, hier = cv2.findContours(close,cv2.RETR_EXTERNAL,cv2.CHAIN_APPROX_SIMPLE)
for cnt in contour:
    x,y,w,h = cv2.boundingRect(cnt)
    if h/w > 5:
        cv2.drawContours(close,[cnt],0,255,-1)
    else:
        cv2.drawContours(close,[cnt],0,0,-1)
close = cv2.morphologyEx(close,cv2.MORPH_CLOSE,None,iterations = 2)
closex = close.copy()

# Finding Horizontal Lines
kernely = cv2.getStructuringElement(cv2.MORPH_RECT,(10,2))
dy = cv2.Sobel(res,cv2.CV_16S,0,2)
dy = cv2.convertScaleAbs(dy)
cv2.normalize(dy,dy,0,255,cv2.NORM_MINMAX)
ret,close = cv2.threshold(dy,0,255,cv2.THRESH_BINARY+cv2.THRESH_OTSU)
close = cv2.morphologyEx(close,cv2.MORPH_DILATE,kernely)

contour, hier = cv2.findContours(close,cv2.RETR_EXTERNAL,cv2.CHAIN_APPROX_SIMPLE)

for cnt in contour:
    x,y,w,h = cv2.boundingRect(cnt)
    if w/h > 5:
        cv2.drawContours(close,[cnt],0,255,-1)
    else:
        cv2.drawContours(close,[cnt],0,0,-1)

close = cv2.morphologyEx(close,cv2.MORPH_DILATE,None,iterations = 2)
closey = close.copy()

# Finding Grid Points
res = cv2.bitwise_and(closex,closey)

# Correcting the defects
contour, hier = cv2.findContours(res,cv2.RETR_LIST,cv2.CHAIN_APPROX_SIMPLE)
centroids = []
for cnt in contour:
    mom = cv2.moments(cnt)
    (x,y) = int(mom['m10']/mom['m00']), int(mom['m01']/mom['m00'])
    cv2.circle(img,(x,y),4,(0,255,0),-1)
    centroids.append((x,y))

centroids = np.array(centroids,dtype = np.float32)
c = centroids.reshape((100,2))
c2 = c[np.argsort(c[:,1])]

b = np.vstack([c2[i*10:(i+1)*10][np.argsort(c2[i*10:(i+1)*10,0])] for i in range(10)])
coord = b.reshape((10,10,2))

#print(coord)
#plt.imshow(img)
#plt.show()

sudoku = np.zeros((9,9))

def crop():
    for i in np.arange(0,9):
        for j in np.arange(0,9):
            #                   (Moves Top Down) (Moves Bottom Up)
            crop_img = img[int(coord[i][j][1])+6:int(coord[i+1][j][1])-6, int(coord[i][j][0])+6:int(coord[i][j+1][0])-6]
            gray = cv2.cvtColor(crop_img,cv2.COLOR_BGR2GRAY)
            adaptive_theshold = cv2.adaptiveThreshold(gray, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, cv2.THRESH_BINARY, 85, 10)
            characters = pytesseract.image_to_string(adaptive_theshold, config = "--psm 10")
            #print(characters)
            a = list(characters)
            if a[0].isnumeric(): sudoku[j,i] = np.around(int(a[0]))

crop()

sudoku = sudoku.astype(int)
grid = sudoku
sudoku = np.rot90(np.fliplr(sudoku))

# Load Fonts
font1 = pygame.font.SysFont("calibri", 30)

def valid(row, column, guess):
	#Checking if Number is in a Particular Row
	for i in range(0, len(sudoku[0])):
		if sudoku[row][i] == guess:
			return False

	#Checking if Number is in a Particular Column
	for i in range(0, len(sudoku[0])):
		if sudoku[i][column] == guess:
			return False

	#Checking if Number is in a Particular Square
	#Take column number and divide by 3 to find the remainder, then muliply by 3.
	#the remainder for the first 3 columns should be 0, the second 3 should be 1, the third 3 should be 2.
	x0 = (column // 3) * 3
	y0 = (row // 3) * 3
	for i in range(0,3):
		for j in range(0,3):
			if sudoku[y0+i][x0+j] == guess:
				return False

	return True

# Solve Sudoku Using Recursion
solved = 0
def solve():
	for row in range(len(sudoku[0])):
		for column in range(len(sudoku[0])):
			if sudoku[row][column] == 0:
				for guess in range(1,10):
					if valid(row,column,guess):
						sudoku[row][column] = guess

						# Print Numbers for Solution on Board
						pygame.draw.rect(screen, (200, 200, 200), ((column)*50 + 50, (row)*50+ 50,51, 51))
						value = font1.render(str(guess), True, (50,130,50))
						screen.blit(value, ((69+(column)*50, 60+(row)*50)))
						pygame.display.update()
						pygame.time.delay(1)

						# Redraw board so that lines stay on top of colors
						for i in range (0,10):
							if(i%3==0):
								pygame.draw.line(screen, (0,0,0),(50+50*i,50),(50+50*i,500), 4)
								pygame.draw.line(screen, (0,0,0),(500,50+50*i),(50, 50+50*i), 4)

							pygame.draw.line(screen, (0,0,0),(50+50*i,50),(50+50*i,500), 2)
							pygame.draw.line(screen, (0,0,0),(500,50+50*i),(50, 50+50*i), 2)
							pygame.display.update()
						solve()

						global solved
						if(solved ==1):
							return

						sudoku[row][column] = 0
				return
	solved = 1

# Initial Screen with Numbers from Input Sudoku
def main():
	pygame.init()
	win = pygame.display.set_mode((WIDTH,WIDTH))
	pygame.display.set_caption("Sudoku Solver (PRESS ENTER TO SOLVE)")
	win.fill(background_color)

	for i in range (9):
		for j in range (9):
			if grid[i][j]!= 0:

				# Fill grey color in already numbered grid
				pygame.draw.rect(screen, (200, 200, 200), (50+50*i, 50+50*j, 51, 51))

				# Fill grid with initial grid numbers specified
				text1 = font1.render(str(grid[i][j]), 1, (0, 0, 0))
				screen.blit(text1, (68+50*i, 50+50*j+12))

	for i in range (0,10):
		if(i%3==0):
			pygame.draw.line(win, (0,0,0),(50+50*i,50),(50+50*i,500), 4)
			pygame.draw.line(win, (0,0,0),(500,50+50*i),(50, 50+50*i), 4)

		pygame.draw.line(win, (0,0,0),(50+50*i,50),(50+50*i,500), 2)
		pygame.draw.line(win, (0,0,0),(500,50+50*i),(50, 50+50*i), 2)
	pygame.display.update()

	while True:
		for event in pygame.event.get():
			if event.type == pygame.QUIT:
				pygame.quit()
				return
		keys = pygame.key.get_pressed()
		if keys[pygame.K_RETURN]:
			solve()
main()
pygame.quit()
cv2.waitKey(0)
