import cv2
import numpy as np

#globals
vert = 0
points = [[-1, -1], [-1, -1], [-1, -1], [-1, -1]]

def sortPoints(points: list, clockwise: bool=True):
    ''' 
    Sort points in clockwise/anti-clockwise order by using polar coordinate system
    points: list of tuples (x,y)
    '''
    #shift points such that centroid is at (0,0)
    points_arr = np.array(points)
    centroid = np.sum(points_arr, axis=0)/points_arr.shape[0]
    points_arr = points_arr - centroid

    x = points_arr[:, 0]
    y = points_arr[:, 1]
    t = np.arctan2(y,x) # polar angle
    r = np.sqrt(x**2+y**2) # radius
    if clockwise:
        return [points[i] for i in np.lexsort((r, t))]
    else:
        return [points[-i] for i in np.lexsort((r, t))]

def draw_per_map(event, x, y, flags, param):
    global vert, points
    line_thickness = 5

    if event == cv2.EVENT_LBUTTONDOWN:
        if vert == 0:
            print("vert 1")
            # x = 514
            # y = 194
            points[0][0] = x
            points[0][1] = y
            cv2.circle(img, (x, y), 5, (0, 0, 255), -1)
            vert += 1
        elif vert == 1:
            print("vert 2")
            # x = 514
            # y = 877
            points[1][0] = x
            points[1][1] = y
            cv2.line(img, (points[0][0], points[0][1]), (x, y), (0, 0, 255), line_thickness)
            vert += 1
        elif vert == 2:
            print("vert 3")
            # x = 853
            # y = 623
            points[2][0] = x
            points[2][1] = y
            cv2.line(img, (points[1][0], points[1][1]), (x, y), (0, 0, 255), line_thickness)
            vert += 1
        elif vert == 3:
            print("vert 4")
            # x = 853
            # y = 402
            points[3][0] = x
            points[3][1] = y
            cv2.line(img, (points[2][0], points[2][1]), (x, y), (0, 0, 255), line_thickness)
            cv2.line(img, (x, y), (points[0][0], points[0][1]), (0, 0, 255), line_thickness)
            vert += 1

            #point assign
            point1 = tuple(points[0])
            point2 = tuple(points[1])
            point3 = tuple(points[2])
            point4 = tuple(points[3])

            #point sort
            point1, point2, point3, point4 = sortPoints([point1, point2, point3, point4])
            
            #vanishing point get
            hor1 = (point1, point2)
            hor2 = (point4, point3)
            vert1 = (point1, point4)
            vert2 = (point2, point3)
            van_hor = line_intersection(hor1, hor2)
            van_vert = line_intersection(vert1, vert2)
            
            grid(point1, point2, point3, point4, line_thickness, 3, van_hor, van_vert)

def grid(point1, point2, point3, point4, line_thickness, recursion_num, van_hor, van_vert):
    van_hor_og = van_hor
    van_vert_og = van_vert

    #center point
    line1_3 = (point1, point3)
    line2_4 = (point2, point4)
    point_o = line_intersection(line1_3, line2_4)
    print("point_o", point_o)
    cvcircle(img, point_o, 5, (0, 255, 0), -1)
    
    #define 4 lines of polygon
    hor1 = (point1, point2)
    hor2 = (point4, point3)
    vert1 = (point1, point4)
    vert2 = (point2, point3)
    
    #no vanshing point 
    if van_hor == None:
        van_hor = line_intersection((point_o, (point4[0], point_o[1])), vert1)
    if van_vert == None:
        van_vert = line_intersection((point_o, (point_o[0], point2[1])), hor1)

    #i1
    left_idx = line_intersection(vert1, (point_o, van_hor))
    print(left_idx, "left_idx")
    cvcircle(img, left_idx, 5, (0, 255, 0), -1)

    #i2
    bot_idx = line_intersection(hor2, (point_o, van_vert))
    print(bot_idx, "bot_idx")
    cvcircle(img, bot_idx, 5, (0, 255, 0), -1)

    #i3
    right_idx = line_intersection(vert2, (point_o, van_hor))
    print(right_idx, "right_idx")
    cvcircle(img, right_idx, 5, (0, 255, 0), -1)

    #i4
    top_idx = line_intersection(hor1, (point_o, van_vert))
    print(top_idx, "top_idx")
    cvcircle(img, top_idx, 5, (0, 255, 0), -1)

    #hor line draw
    cvline(img, left_idx, right_idx, (0, 255, 0), line_thickness)
    #vert line draw
    cvline(img, top_idx, bot_idx, (0, 255, 0), line_thickness)

    van_hor = van_hor_og
    van_vert = van_vert_og
    
    recursion_num -= 1
    if recursion_num > 0:
        grid(point1, top_idx, point_o, left_idx, line_thickness, recursion_num, van_hor, van_vert)
        grid(top_idx, point2, right_idx, point_o, line_thickness, recursion_num, van_hor, van_vert)
        grid(point_o, right_idx, point3, bot_idx, line_thickness, recursion_num, van_hor, van_vert)
        grid(left_idx, point_o, bot_idx, point4, line_thickness, recursion_num, van_hor, van_vert)

def line_intersection(line1, line2):
    xdiff = (line1[0][0] - line1[1][0], line2[0][0] - line2[1][0])
    ydiff = (line1[0][1] - line1[1][1], line2[0][1] - line2[1][1])

    def det(a, b):
        return a[0] * b[1] - a[1] * b[0]

    div = det(xdiff, ydiff)
    if div == 0:
        return None

    d = (det(*line1), det(*line2))
    x = det(d, xdiff) / div
    y = det(d, ydiff) / div
    return (x, y)

#calls cv2.line() but rounds tuples
def cvline(image, xy1, xy2, color, thickness):
    cv2.line(image, rou(xy1), rou(xy2), color, thickness)

#calls cv2.line() but rounds tuples
def cvcircle(image, xy1, radius, color, thickness):
    lmao = 1
    # cv2.circle(image, rou(xy1), radius, color, thickness)

#rounds xy tuple
def rou(xy_tup):
    return (round(xy_tup[0]), round(xy_tup[1]))

if __name__ == '__main__':
    img = cv2.imread('1080p.png')

    cv2.namedWindow('window', cv2.WINDOW_NORMAL)
    cv2.setMouseCallback('window', draw_per_map)

    copy = img.copy()
    while True:
        cv2.imshow('window', img)

        #reset
        if cv2.waitKey(1) & 0xFF == ord('r'):
            print("reset")
            img = copy.copy()
            vert = 0

        if cv2.waitKey(1) & 0xFF == ord('q'):
            break
    
    cv2.destroyAllWindows()
