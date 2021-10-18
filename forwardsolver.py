import sys
from copy import deepcopy
from PIL import Image
import pygame
import cv2
import numpy as np
import matplotlib.pyplot as plt
from itertools import product
from pytesseract import *
pytesseract.tesseract_cmd = 'C:\\Program Files\\Tesseract-OCR\\tesseract.exe'

# Initialise Pygame
pygame.init()
pygame.font.init()

# Load Fonts
font1 = pygame.font.SysFont("calibri", 30)

# Defining Values
WIDTH = 550
background_color = (251,247,245)
screen = pygame.display.set_mode((500, 600))
x = 0
y = 0

img = cv2.imread('sudoku4.jpeg')
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
            #crop_img = img[int(coord[i][j][0])+6:int(coord[i][j+1][0])-10, int(coord[i][j][1])+6:int(coord[i+1][j][1])-6]
            #crop_img = img[int(coord[i][j][0]):int(coord[i][j+1][0]), int(coord[i][j][1]):int(coord[i+1][j][1])]
            crop_img = img[int(coord[i][j][1])+6:int(coord[i+1][j][1])-6, int(coord[i][j][0])+6:int(coord[i][j+1][0])-6]
            #cv2.imshow('crop'+str(i)+str(j),crop_img)
            #plt.imshow(crop_img)
            gray = cv2.cvtColor(crop_img,cv2.COLOR_BGR2GRAY)
            adaptive_theshold = cv2.adaptiveThreshold(gray, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, cv2.THRESH_BINARY, 85, 10)
            characters = pytesseract.image_to_string(adaptive_theshold, config = "--psm 10")
            #print(characters)
            a = list(characters)
            if a[0].isnumeric(): sudoku[j,i] = np.around(int(a[0]))

crop()

sudoku = sudoku.astype(int)
grid = sudoku
sudoku1 = np.rot90(np.fliplr(sudoku))
field = sudoku1.tolist()

def output(a):
    sys.stdout.write(str(a))

def print_field(field):
    if not field:
        output("No solution")
        return
    for i in range(9):
        for j in range(9):
            cell = field[i][j]
            if cell == 0 or isinstance(cell, set):
                output('.')
            else:
                output(cell)
            if (j + 1) % 3 == 0 and j < 8:
                output(' |')

            if j != 8:
                output(' ')
        output('\n')
        if (i + 1) % 3 == 0 and i < 8:
            output("- - - + - - - + - - -\n")

def read(field):
    """ Read field into state (replace 0 with set of possible values) """

    state = deepcopy(field)
    for i in range(9):
        for j in range(9):
            cell = state[i][j]
            if cell == 0:
                state[i][j] = set(range(1,10))

    return state

state = read(field)


def done(state):
    """ Are we done? """

    for row in state:
        for cell in row:
            if isinstance(cell, set):
                return False
    return True


def propagate_step(state):
    """
    Propagate one step.

    @return:  A two-tuple that says whether the configuration
              is solvable and whether the propagation changed
              the state.
    """

    new_units = False

    # propagate row rule
    for i in range(9):
        row = state[i]
        values = set([x for x in row if not isinstance(x, set)])
        for j in range(9):
            if isinstance(state[i][j], set):
                state[i][j] -= values
                if len(state[i][j]) == 1:
                    val = state[i][j].pop()
                    state[i][j] = val
                    values.add(val)
                    new_units = True
                elif len(state[i][j]) == 0:
                    return False, None

    # propagate column rule
    for j in range(9):
        column = [state[x][j] for x in range(9)]
        values = set([x for x in column if not isinstance(x, set)])
        for i in range(9):
            if isinstance(state[i][j], set):
                state[i][j] -= values
                if len(state[i][j]) == 1:
                    val = state[i][j].pop()
                    state[i][j] = val
                    values.add(val)
                    new_units = True
                elif len(state[i][j]) == 0:
                    return False, None

    # propagate cell rule
    for x in range(3):
        for y in range(3):
            values = set()
            for i in range(3 * x, 3 * x + 3):
                for j in range(3 * y, 3 * y + 3):
                    cell = state[i][j]
                    if not isinstance(cell, set):
                        values.add(cell)
            for i in range(3 * x, 3 * x + 3):
                for j in range(3 * y, 3 * y + 3):
                    if isinstance(state[i][j], set):
                        state[i][j] -= values
                        if len(state[i][j]) == 1:
                            val = state[i][j].pop()
                            state[i][j] = val
                            values.add(val)
                            new_units = True
                        elif len(state[i][j]) == 0:
                            return False, None

    return True, new_units

def propagate(state):
    """ Propagate until we reach a fixpoint """
    while True:
        solvable, new_unit = propagate_step(state)
        if not solvable:
            return False
        if not new_unit:
            return True


def solve(state):
    """ Solve sudoku """

    solvable = propagate(state)

    if not solvable:
        return None

    if done(state):
        return state

    for i in range(9):
        for j in range(9):
            cell = state[i][j]
            if isinstance(cell, set):
                for value in cell:
                    new_state = deepcopy(state)
                    new_state[i][j] = value
                    solved = solve(new_state)
                    if solved is not None:
                        return solved
                return None

guess = np.rot90(np.fliplr(solve(state)))

# Initial Screen with Numbers from Input Sudoku
def main():
    pygame.init()
    win = pygame.display.set_mode((WIDTH,WIDTH))
    pygame.display.set_caption("Sudoku")
    win.fill(background_color)

	# def result():
	# 	text1 = font1.render("FINISHED PRESS R or D", 1, (0, 0, 0))
	# 	screen.blit(text1, (20, 570))
    for i in range (9):
        for j in range (9):
            if grid[i][j]!= 0:
                # Fill grey color in already numbered grid
                pygame.draw.rect(screen, (200, 200, 200), (50+50*i, 50+50*j, 51, 51))
                # Fill grid with initial grid numbers specified
                text1 = font1.render(str(grid[i][j]), 1, (0, 0, 0))
                screen.blit(text1, (68+50*i, 50+50*j+12))
            else:
                pygame.draw.rect(screen, (200, 200, 200), ((i)*50 + 50, (j)*50+ 50,51, 51))
                value = font1.render(str(guess[i][j]), True, (50,130,50))
                screen.blit(value, ((69+(i)*50, 60+(j)*50)))

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
            solve(state)
main()
pygame.quit()
cv2.waitKey(0)

#print(solve(state))
#print_field(solve(state))
