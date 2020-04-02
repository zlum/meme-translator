import cv2

class Point: 
    def __init__(self, x, y): 
        self.x = x 
        self.y = y 

# Returns true if two rectangles(l1, r1)  
# and (l2, r2) overlap 
def doOverlap(l1, r1, l2, r2): 
	# If one rectangle is on left side of other 
    if(l1.x > r2.x or l2.x > r1.x):
        return False
	# If one rectangle is above other 
    if(l1.y > r2.y or l2.y > r1.y):
        return False

    return True

# Merges all overlapping rectangles
def merge(rects):
    combined = [] # Indexes of combined rectangles
    multirects = [] # List of combined rectangles

    restart = True

    i = 0
    for (startX1, startY1, endX1, endY1) in rects:
        if i in combined:
            i = i + 1
            continue

        combined.append(i)

        while restart:
            j = 0
            restart = False
            neverOverlap = True

            for (startX2, startY2, endX2, endY2) in rects:
                if j in combined: 
                    j = j + 1
                    continue
                
                if doOverlap(Point(startX1, startY1), Point(endX1, endY1), Point(startX2, startY2), Point(endX2, endY2)):
                    combined.append(j)

                    startX1 = min(startX1, startX2)
                    startY1 = min(startY1, startY2)
                    endX1 = max(endX1, endX2)
                    endY1 = max(endY1, endY2)

                    neverOverlap = False
                    restart = True
                    break

                j = j + 1

            if neverOverlap == True:
                multirects.append((startX1, startY1, endX1, endY1))
                restart = True
                break

        i = i + 1

    return multirects
