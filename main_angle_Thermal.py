import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
import cv2
import os




""" Insert Inputs: 
    1- Location of the Image and Data"""  
    
path = 'September 16'
clock = '12'





""" 1 - Load Image """
image_name = 'Thermal.png'
data_name = 'Thermal.csv'
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
plt.title('Thermal image from Camera')
plt.colorbar()
plt.show()




""" 2- Load Temprature data from CSV file """
Dataname = os.path.join(Folder1, Folder2, path, clock, data_name)  
Temprature_data_F = pd.read_csv(Dataname, sep=',', header=0, encoding='utf-8').to_numpy() ## Fahrenheit
Temprature_data = (Temprature_data_F - 32)/1.8 ## Convert to Centigrade

plt.figure()
plt.imshow(Temprature_data)
plt.title('Temprature (°C) Map from csv')
plt.colorbar()
plt.show()





""" 3- Do Angle Correction for Thermal Image  """
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
corners[0] = output[3][1:][0]
corners[1] = output[3][1:][1]
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

h_img, w_img = image.shape[:2]
H, _ = cv2.findHomography(corners, dst, method=cv2.RANSAC, ransacReprojThreshold=3.0)
float_formatter = "{:.2f}".format
np.set_printoptions(formatter={'float_kind': float_formatter})
print('\nThe homography matrix is: \n', H)

## Unwarp the original Image
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
plt.show()
# plt.savefig('angle correction for image.jpg',dpi=300)


""" 3- Do Angle Correction for Thermal CVS file """

## Unwarp the Thermal csv data
unwarp_cvs_rgb = cv2.warpPerspective(Temprature_data, H, (w_dst, h_dst), flags=cv2.INTER_LINEAR)

# plot
f, (ax1, ax2) = plt.subplots(1, 2)
ax1.imshow(Temprature_data)
x = [corners[0][0], corners[2][0], corners[3][0], corners[1][0], corners[0][0]]
y = [corners[0][1], corners[2][1], corners[3][1], corners[1][1], corners[0][1]]
ax1.plot(x, y, color='yellow', linewidth=3)
ax1.set_ylim([h_img, 0])
ax1.set_xlim([0, w_img])
ax1.set_title('Targeted Area in Original Image')
ax2.imshow(unwarp_cvs_rgb)
ax2.set_ylim([h_dst, 0])
ax2.set_xlim([0, w_dst])
ax2.set_xticks([0, int((w_dst-1)/2), w_dst-1]) 
ax2.set_yticks([0, int((h_dst-1)/2), h_dst-1]) 
ax2.set_title('Unwarped Image')
image_name2 = 'Ang Corr Thermal.png'
onja = os.path.join(Folder1, path_save, path, clock, image_name2) 
plt.savefig(onja,dpi=300)


plt.figure()
plt.imshow(unwarp_cvs_rgb)
plt.title('csv angle corrected')
plt.colorbar()
onja2 = os.path.join(Folder1, path_save, path, clock, image_name) 
plt.savefig(onja2,dpi=300)

data_name2 = 'Thermal.txt'
onja3 = os.path.join(Folder1, path_save, path, clock, data_name2) 
np.savetxt(onja3, unwarp_cvs_rgb)



