import cv2
import numpy as np
import random


# Check to see if a point is inside a rectangle
def rect_contains(rect, point) :
    if point[0] < rect[0] :
        return False
    elif point[1] < rect[1] :
        return False
    elif point[0] > rect[2] :
        return False
    elif point[1] > rect[3] :
        return False
    return True

# Draw a point
def draw_points(img, p, color ) :
    cv2.circle( img, p, 2, color, -1)

# Draw delaunay triangles
def draw_delaunay(img, subdiv, delaunay_color ) :

    triangleList = subdiv.getTriangleList();
    size = img.shape
    r = (0, 0, size[1], size[0])

    for t in triangleList :
        
        pt1 = (t[0], t[1])
        pt2 = (t[2], t[3])
        pt3 = (t[4], t[5])
        
        if rect_contains(r, pt1) and rect_contains(r, pt2) and rect_contains(r, pt3) :
        
            cv2.line(img, pt1, pt2, delaunay_color, 1)
            cv2.line(img, pt2, pt3, delaunay_color, 1)
            cv2.line(img, pt3, pt1, delaunay_color, 1)


# Draw voronoi diagram
def draw_voronoi(img, subdiv) :

    ( facets, centers) = subdiv.getVoronoiFacetList([])

    for i in range(0,len(facets)) :
        ifacet_arr = []
        for f in facets[i] :
            ifacet_arr.append(f)
        
        ifacet = np.array(ifacet_arr, np.int)
        color = (random.randint(0, 255), random.randint(0, 255), random.randint(0, 255))

        cv2.fillConvexPoly(img, ifacet, color);
        ifacets = np.array([ifacet])
        cv2.polylines(img, ifacets, True, (0, 0, 0), 1)
        cv2.circle(img, (centers[i][0], centers[i][1]), 3, (0, 0, 0), -1)
    

if __name__ == '__main__':
    
    # Define the window names
    win_delaunay = "Delaunay Triangulation"
    win_voronoi = "Voronoi Tesselation"
    
    animate = True
    
    delaunay_color = (255,255,255) # white
    points_color = (0,0,0) # black
    
    # Reading the image
    img = cv2.imread("obama.jpg")
    
    # making a copy of the original image
    img_org = img.copy()
    
    size = img.shape
    rect = (0, 0, size[1], size[0])
    
    # creating an instance of Subdiv2D 
    subdiv = cv2.Subdiv2D(rect)
    
    # creating an array of points
    points = []
    
    with open("obama.txt") as fp:
        for line in fp:
            x, y = line.split()
            points.append((int(x), int(y)))
            
    # insert points into subdiv
    for p in points:
        subdiv.insert(p)
        
        # Show animation
        if animate:
            img_copy = img_org.copy()
            
            # Draw the delaunay triangles
            draw_delaunay(img_copy, subdiv, (255, 255, 255))
            
            cv2.imshow(win_delaunay, img_copy)
            cv2.waitKey(100)
            
    # Draw the delaunay triangles
    draw_delaunay(img, subdiv, (255, 255, 255))
    
    # Draw points
    for p in points:
        draw_points(img, p , (0,0,255))
        
    # Allocate space for voronoi diagrams
    img_voronoi = np.zeros(img.shape, dtype= img.dtype)
    
    # draw voronoi diagram
    draw_voronoi(img_voronoi, subdiv)
    
    # show results
    cv2.imshow(win_delaunay, img)
    cv2.imshow(win_voronoi, img_voronoi)
    cv2.waitKey(0)
            
            
            
        
        