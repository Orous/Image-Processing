import numpy as np
import cv2
import matplotlib.pyplot as plt
import os




""" Insert Inputs: 
    1- Location of the Image 2- Height and Width of the rectangle in real life """ 
        
path = 'September 21'
clock = '9'




""" 1 - Load Image """
image_name = 'RGB.png'
Folder1 = 'Data'
Folder2 = 'Modified'
path_save = 'Results'

h_rct = 180 
w_rct = 226
Filename = os.path.join(Folder1, Folder2, path, clock, image_name)  
image_bgr = cv2.imread(Filename) 
image = cv2.cvtColor(image_bgr, cv2.COLOR_BGR2RGB)

# RGB (color) imagery is similar to viewing a digital photograph taken from a plane
plt.figure()
plt.imshow(image)
plt.title('Original image with dots')
plt.colorbar()
plt.show()
# plt.savefig('Original image with dots.jpg',dpi=300)





""" 2 - dot_coordinate_detection: 
    Detecting coordinates of dots of specific color/pixel value
    Inputs:
        image: ndarray. Original Image in RGB format
        lb: ndarray. Lower bound of specific color/pixel value
        ub: ndarray. Upper bound of specific color/pixel value
    Returns:
        centroids: ndarray. Coordinates of dots of assigned color/pixel value """

## Define color of investigated dots.
## These are the RGB values of dots that we added it to the image previously. 
## Since we used the same color for all 4 dots, we have same RGB values for all 4 dots. 
## lb could be the same as RGB values and ub is just 1 value more than lb.         

#lb = np.array([228, 39, 42])     
#ub = np.array([229, 40, 43])

lb = np.array([63, 72, 204])     
ub = np.array([64, 73, 205])


mask = cv2.inRange(image, lb, ub)
plt.imshow(mask)
plt.title('Mask Image')
plt.show()

connectivity = 4
# Perform the operation
output = cv2.connectedComponentsWithStats(mask, connectivity, cv2.CV_32S)
# Get the number of the dots
num_labels = output[0]-1  # output[0] = num_labels + background.
# Get the centroids/coordinates of the dots

""" Two Ways to find Corners  """

" Method1:"
# corners = output[3][1:]
# corners = np.around(corners)  # Around the number before converting to int
# corners = corners.astype(int)  # Convert to int

" Method2:"
corners = np.zeros((4,2))
corners[0] = output[3][1:][1]
corners[1] = output[3][1:][0]
corners[2] = output[3][1:][2]
corners[3] = output[3][1:][3]
corners = np.around(corners)  # Around the number before converting to int
corners = corners.astype(int)  # Convert to int

# print results
print('number of dots, should be 4:',num_labels )
print('The corner points are:')
print('A :', corners[0])
print('B :', corners[1])
print('C :', corners[2])
print('D :', corners[3])

## The main output from these lines is: corners



""" 3- get_destination_points:
     -Get destination points from corners of warped images
    -Approximating height and width of the rectangle: we take maximum of the 2 widths and 2 heights
    Inputs:
        corners: list. which is the main output of previous step. 
        h_rct: Height of the rectangle in real life
        w_rct: Weight of the rectangle in real life
    Returns:
        destination_corners: list
        height: int
        width: int
    """



""" Field Dimensions 
    Top (Single) Small Grass Rectangle: H = 60 cm , W = 115 cm 
    Whole Box Rectangle: H = 360 cm , W = 250 cm """  

 
## If you consider the top small grass rectangle area: 
#h_rct = 60  
#w_rct = 115


## If you consider the big box (half of the box)
#h_rct = 180 ## H is the half of the height of the box  
#w_rct = 250



## If you consider the experiment (Refer to "RGB with points" Figure)
#h_rct = 180 ## H is the half of the height of the box  
#w_rct = 225



w1 = np.sqrt((corners[0][0] - corners[1][0]) ** 2 + (corners[0][1] - corners[1][1]) ** 2)
w2 = np.sqrt((corners[2][0] - corners[3][0]) ** 2 + (corners[2][1] - corners[3][1]) ** 2)
w_dst = max(int(w1), int(w2))

#h1 = np.sqrt((corners[0][0] - corners[2][0]) ** 2 + (corners[0][1] - corners[2][1]) ** 2) ##?? 
#h2 = np.sqrt((corners[1][0] - corners[3][0]) ** 2 + (corners[1][1] - corners[3][1]) ** 2)
#h_dst = max(int(h1), int(h2))
h_dst = int(w_dst/w_rct*h_rct) 

## destination_corners
dst = np.float32([(0, 0), (w_dst - 1, 0), (0, h_dst - 1), (w_dst - 1, h_dst - 1)]) 

print('\nThe destination points are:')
for index, c in enumerate(dst):
    character = chr(65 + index) + "'"
    print(character, ':', c.astype(int))

print('\nThe approximated height and width of the original image is: \n', (h_dst, w_dst))

## The main outputs from these lines are: dst, h_dst, w_dst




""" 4- unwarp:
    Inputs:
        image: np.array
        corners: list
        dst: list. which is the main output of the previous step. 
        h_dst . which is the main output of the previous step.
        w_dst . which is the main output of the previous step.
    Returns:
        un_warped: np.array
    """

h_img, w_img = image.shape[:2]
H, _ = cv2.findHomography(corners, dst, method=cv2.RANSAC, ransacReprojThreshold=3.0)
float_formatter = "{:.2f}".format
np.set_printoptions(formatter={'float_kind': float_formatter})
print('\nThe homography matrix is: \n', H)

## Unwarp the Image
unwarp_img_rgb = cv2.warpPerspective(image, H, (w_dst, h_dst), flags=cv2.INTER_LINEAR)

# plot
f, (ax1, ax2) = plt.subplots(1, 2)
ax1.imshow(image)
x = [corners[0][0], corners[2][0], corners[3][0], corners[1][0], corners[0][0]]
y = [corners[0][1], corners[2][1], corners[3][1], corners[1][1], corners[0][1]]
ax1.plot(x, y, color='yellow', linewidth=3)
ax1.set_ylim([h_img, 0])
ax1.set_xlim([0, w_img])
ax1.set_title('Targeted Area in Original Image')
ax2.imshow(unwarp_img_rgb)
ax2.set_ylim([h_dst, 0])
ax2.set_xlim([0, w_dst])
ax2.set_xticks([0, int((w_dst-1)/2), w_dst-1]) 
ax2.set_yticks([0, int((h_dst-1)/2), h_dst-1]) 
ax2.set_title('Unwarped Image')
# plt.show()
image_name2 = 'Ang Corr RGB.png'
onja = os.path.join(Folder1, path_save, path, clock, image_name2) 
plt.savefig(onja,dpi=300)

## The main outputs from these lines are: unwarp_img_rgb, H


""" Convert the unwarped RGB image to BGR image and save it as BGR:
    If you want to show the corrected image in presentation or if you want to save it and use it later 
    it's better to save as BGR, But, if you want to use the image immidiately after angel correction
    and you need the image pixel values, use RGB one to see the RGB values"""
unwarp_img_bgr = cv2.cvtColor(unwarp_img_rgb, cv2.COLOR_RGB2BGR)  # Convert to BGR for saving the image
Filename_save1 = os.path.join(Folder1, path_save, path, clock, image_name) 
cv2.imwrite(Filename_save1,unwarp_img_bgr)



