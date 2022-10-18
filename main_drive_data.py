import numpy as np
import cv2
import matplotlib.pyplot as plt
import os
from find_sensor_loc_func import find_pix_points, find_pix_points_temp, rec_around_sensor




""" 1 - Load Images """

image_name1 = 'RGB.png'
image_name2 = 'NIR.png'
image_name3 = 'Thermal.txt'
Folder1 = 'Data'
Folder2 = 'Results'


Filename1 = os.path.join(Folder1, Folder2, image_name1)  
RGB_image_bgr = cv2.imread(Filename1) 
RGB_image = cv2.cvtColor(RGB_image_bgr, cv2.COLOR_BGR2RGB)

Filename2 = os.path.join(Folder1, Folder2,  image_name2)  
NIR_image_bgr = cv2.imread(Filename2) 
NIR_image = cv2.cvtColor(NIR_image_bgr, cv2.COLOR_BGR2RGB)

Filename3 = os.path.join(Folder1, Folder2,  image_name3)  
Thermal_csv = np.loadtxt(Filename3)


# RGB Image
plt.figure()
plt.imshow(RGB_image)
plt.title('RGB Image')
plt.colorbar()
plt.show()


# NIR Image
plt.figure()
plt.imshow(NIR_image)
plt.title('NIR Image')
plt.colorbar()
plt.show()


# Thermal Image
plt.figure()
plt.imshow(Thermal_csv)
plt.title('Thermal Image')
plt.colorbar()
plt.show()








""" 2- Specify RED, NIR, and LST """

red = RGB_image[:, :, 0].astype(np.float64)  
nir = NIR_image[:, :, 0].astype(np.float64)
lst = Thermal_csv










""" 3- Load the pixels of the sensor locations for each image"""

pix_points_RGB = find_pix_points(RGB_image)
pix_points_NIR = find_pix_points(NIR_image)
pix_points_Thermal = find_pix_points_temp(Thermal_csv)





""" 4- Obtain RED, NIR, LST at the Sensor locations Rectangles"""

[RGB_increment_x , RGB_increment_y] = rec_around_sensor(RGB_image)
[NIR_increment_x , NIR_increment_y] = rec_around_sensor(NIR_image)
[Thermal_increment_x , Thermal_increment_y] = rec_around_sensor(Thermal_csv)


red_sample=[]
for i in range(len(pix_points_RGB)):
    red_sample.append(np.mean(red[int(pix_points_RGB[i,1]-RGB_increment_y/2):int(pix_points_RGB[i,1]+RGB_increment_y/2),
                                  int(pix_points_RGB[i,0]-RGB_increment_x/2):int(pix_points_RGB[i,0]+RGB_increment_x/2)]))


nir_sample=[]
for i in range(len(pix_points_NIR)):
    nir_sample.append(np.mean(nir[int(pix_points_NIR[i,1]-NIR_increment_y/2):int(pix_points_NIR[i,1]+NIR_increment_y/2),
                                  int(pix_points_NIR[i,0]-NIR_increment_x/2):int(pix_points_NIR[i,0]+NIR_increment_x/2)]))

   
lst_sample=[]
for i in range(len(pix_points_Thermal)):
    lst_sample.append(np.mean(lst[int(pix_points_Thermal[i,1]-Thermal_increment_y/2):int(pix_points_Thermal[i,1]+Thermal_increment_y/2),
                                  int(pix_points_Thermal[i,0]-Thermal_increment_x/2):int(pix_points_Thermal[i,0]+Thermal_increment_x/2)]))
    
    
    




""" 5- NDVI: Normalized Difference Vegetative Index  """

red_sample_array = np.array(red_sample)
nir_sample_array = np.array(nir_sample)
lst_sample_array = np.array(lst_sample)

ndvi_sample = np.divide((nir_sample_array - red_sample_array), (nir_sample_array + red_sample_array))
 





""" 6- TVDI: Temperature Vegetation Dryness Index """

lst_zone1 = lst[ 0:int(np.shape(lst)[0]/2) , 0:int(np.shape(lst)[1]/2) ]    
lst_zone2 = lst[ 0:int(np.shape(lst)[0]/2) , int(np.shape(lst)[1]/2):np.shape(lst)[1] ]    
lst_zone3 = lst[ int(np.shape(lst)[0]/2):np.shape(lst)[0] , 0:int(np.shape(lst)[1]/2) ]    
lst_zone4 = lst[ int(np.shape(lst)[0]/2):np.shape(lst)[0] , int(np.shape(lst)[1]/2):np.shape(lst)[1] ]    
Tsmax = np.array( [ np.max(lst_zone1) , np.max(lst_zone2) , np.max(lst_zone3) , np.max(lst_zone4) ] )
Tsmin = np.array( [ np.min(lst_zone1) , np.min(lst_zone2) , np.min(lst_zone3) , np.min(lst_zone4) ] )

ndvi_mean_z1 = np.mean([ ndvi_sample[8] , ndvi_sample[9] , ndvi_sample[14] , ndvi_sample[15] ])
ndvi_mean_z2 = np.mean([ ndvi_sample[10] , ndvi_sample[11] , ndvi_sample[12] , ndvi_sample[13] ])
ndvi_mean_z3 = np.mean([ ndvi_sample[0] , ndvi_sample[1] , ndvi_sample[6] , ndvi_sample[7] ])
ndvi_mean_z4 = np.mean([ ndvi_sample[2] , ndvi_sample[3] , ndvi_sample[4] , ndvi_sample[6] ])
ndvi_mean = np.array( [ ndvi_mean_z1 , ndvi_mean_z2 , ndvi_mean_z3 , ndvi_mean_z4 ] )


""" Method 1: Fit Polynomial for Tmax adn Tmin """
coef = np.polyfit(ndvi_mean, Tsmax, 1)
coef1 = np.polyfit(ndvi_mean, Tsmin, 1)
# maxm_fit = np.polyval(coef, ndvi_mean)
tvdi_sample1 = np.divide(( lst_sample_array - (np.multiply(coef1[0], ndvi_sample) + coef1[1]) ), 
                  ( (np.multiply(coef[0], ndvi_sample) + coef[1]) - (np.multiply(coef1[0], ndvi_sample) + coef1[1]) ))



""" Method 2: Fix Tmin and Fit Polynomial for Tmax """
coeff = np.polyfit(ndvi_mean, Tsmax, 1)
# maxm_fit = np.polyval(coef, ndvi_mean)
tvdi_sample2 = np.divide(( lst_sample_array - np.min(lst) ), ( (np.multiply(coeff[0], ndvi_sample) + coeff[1]) - np.min(lst)))



""" Method 3: Fix Tmin and Tmax """
tvdi_sample3 = np.divide((lst_sample_array - np.min(lst)), (np.max(lst) - np.min(lst)))








    
""" 7- Collect All Data Together """
All_Data = np.array( [ red_sample , nir_sample , lst_sample , ndvi_sample, tvdi_sample1, tvdi_sample2, tvdi_sample3 ] ).T
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    


























































