import numpy as np
import cv2
import matplotlib.pyplot as plt
import os





def find_pix_points(image):

    """ Find Sensor Location """ 
    h_rct = 180 
    w_rct = 226
    
    delta_x = (w_rct/image.shape[1])  # m 
    delta_y = (h_rct/image.shape[0])  # m
    
    c = tuple((int(image.shape[1]/2),int(image.shape[0]/2))) # center of the field
    
    ## Correct the center location
    # c1 = np.zeros((2,))
    # c1[0] = c[1]-90
    # c1[1] = c[0]+100
    
    
    # L_c=[]
    # L_c = cv2.circle(image, (int(c[0]),int(c[1])), 40, (0,0,255), -1)           
    # plt.figure()
    # plt.imshow(L_c)
    # plt.title('Location of the center')
    # plt.colorbar()
    # plt.show()
          
    
            ## Define the Location of the Flags
    L={0:(-100,50) , 1:(-50,50) , 2:(0,50) , 3:(50,50) , 4:(100,50) , 
       5:(100,0) , 6:(50,0) , 7:(0,0) , 8:(-50,0) , 9:(-100,0) , 
       10:(-100,-50) , 11:(-50,-50) , 12:(0,-50) , 13:(50,-50) , 14:(100,-50) }  
    
    
        ## Get the Pixel of the Points
    pix_flags = np.zeros((len(L),2))
    for i in range(len(L)):
        pix_flags[i,0] = round(int(c[0])+(L[i][0]/delta_x))
        pix_flags[i,1] = round(int(c[1])+(L[i][1]/delta_y)) 
        
    
    pix_flags_cm = np.zeros((len(L),2))
    for i in range(len(L)):
        pix_flags_cm[i,0] = round((pix_flags[i,0] * w_rct) / image.shape[1])
        pix_flags_cm[i,1] = round(((image.shape[0] - pix_flags[i,1]) * h_rct) / image.shape[0])
    
    
    # flag_location=[]
    # for i in range(len(pix_flags)):
    #         flag_location = cv2.circle(image, (int(pix_flags[i,0]),int(pix_flags[i,1])), 40, (255,0,0), -1)         
    # plt.figure()
    # plt.imshow(flag_location)
    # plt.title('Flag locations')
    # plt.colorbar()
    # plt.show()
    # plt.savefig('Flag locations.jpg',dpi=300)
    
    # flag_location_rgb = cv2.cvtColor(flag_location, cv2.COLOR_RGB2BGR)  # Convert to BGR for saving the image
    # path_circle = 'Data\Flag locations.png'
    # cv2.imwrite(path_circle,flag_location_rgb)
    
    
    ## Define the Location of the Sensors
    S={0:(-75,70) , 1:(-25,70) , 2:(25,70) , 3:(75,70) , 
       4:(75,25) , 5:(25,25) , 6:(-25,25) , 7:(-75,25) ,
       8:(-75,-25) , 9:(-25,-25) , 10:(25,-25) , 11:(75,-25) , 
       12:(75,-70) , 13:(25,-70) , 14:(-25,-70) , 15:(-75,-70)}  
    
    
    pix_points = np.zeros((len(S),2))
    for i in range(len(S)):
        pix_points[i,0] = round(int(c[0])+(S[i][0]/delta_x))
        pix_points[i,1] = round(int(c[1])+(S[i][1]/delta_y)) 
    
    
    pix_points_cm = np.zeros((len(S),2))
    for i in range(len(S)):
        pix_points_cm[i,0] = round((pix_points[i,0] * w_rct) / image.shape[1])
        pix_points_cm[i,1] = round(( (image.shape[0] - pix_points[i,1]) * h_rct) / image.shape[0])
    
    
    # sensor_location=[]
    # for i in range(len(pix_points)):
    #         sensor_location = cv2.circle(image, (int(pix_points[i,0]),int(pix_points[i,1])), 40, (0,0,255), -1)         
    # plt.figure()
    # plt.imshow(sensor_location)
    # plt.title('Sensor locations')
    # plt.colorbar()
    # plt.show()
    # plt.savefig('Sensor locations.jpg',dpi=300)
    # sensor_location_rgb = cv2.cvtColor(sensor_location, cv2.COLOR_RGB2BGR)  # Convert to BGR for saving the image
    # path_circle = 'Data\Sensor locations.png'
    # cv2.imwrite(path_circle,sensor_location_rgb)
    
    return pix_points











def find_pix_points_temp(image):

    """ Find Sensor Location """ 
    h_rct = 180 
    w_rct = 226
    
    delta_x = (w_rct/image.shape[1])  # m 
    delta_y = (h_rct/image.shape[0])  # m
    
    c = tuple((int(image.shape[1]/2),int(image.shape[0]/2))) # center of the field
    
    ## Correct the center location
    # c1 = np.zeros((2,))
    # c1[0] = c[1]-90
    # c1[1] = c[0]+100
    
    
    # L_c=[]
    # L_c = cv2.circle(image, (int(c[0]),int(c[1])), 4, (33,142,140), -1)           
    # plt.figure()
    # plt.imshow(L_c)
    # plt.title('Location of the center')
    # plt.colorbar()
    # plt.show()
          
    
            ## Define the Location of the Flags
    L={0:(-100,50) , 1:(-50,50) , 2:(0,50) , 3:(50,50) , 4:(100,50) , 
       5:(100,0) , 6:(50,0) , 7:(0,0) , 8:(-50,0) , 9:(-100,0) , 
       10:(-100,-50) , 11:(-50,-50) , 12:(0,-50) , 13:(50,-50) , 14:(100,-50) }  
    
    
        ## Get the Pixel of the Points
    pix_flags = np.zeros((len(L),2))
    for i in range(len(L)):
        pix_flags[i,0] = round(int(c[0])+(L[i][0]/delta_x))
        pix_flags[i,1] = round(int(c[1])+(L[i][1]/delta_y)) 
        
    
    pix_flags_cm = np.zeros((len(L),2))
    for i in range(len(L)):
        pix_flags_cm[i,0] = round((pix_flags[i,0] * w_rct) / image.shape[1])
        pix_flags_cm[i,1] = round(((image.shape[0] - pix_flags[i,1]) * h_rct) / image.shape[0])
    
    
    # flag_location=[]
    # for i in range(len(pix_flags)):
    #         flag_location = cv2.circle(image, (int(pix_flags[i,0]),int(pix_flags[i,1])), 4, (33,142,140), -1)         
    # plt.figure()
    # plt.imshow(flag_location)
    # plt.title('Flag locations')
    # plt.colorbar()
    # plt.show()
    # plt.savefig('Flag locations.jpg',dpi=300)
    
    # flag_location_rgb = cv2.cvtColor(flag_location, cv2.COLOR_RGB2BGR)  # Convert to BGR for saving the image
    # path_circle = 'Data\Flag locations.png'
    # cv2.imwrite(path_circle,flag_location_rgb)
    
    
    ## Define the Location of the Sensors
    S={0:(-75,70) , 1:(-25,70) , 2:(25,70) , 3:(75,70) , 
       4:(75,25) , 5:(25,25) , 6:(-25,25) , 7:(-75,25) ,
       8:(-75,-25) , 9:(-25,-25) , 10:(25,-25) , 11:(75,-25) , 
       12:(75,-70) , 13:(25,-70) , 14:(-25,-70) , 15:(-75,-70)}  
    
    
    pix_points = np.zeros((len(S),2))
    for i in range(len(S)):
        pix_points[i,0] = round(int(c[0])+(S[i][0]/delta_x))
        pix_points[i,1] = round(int(c[1])+(S[i][1]/delta_y)) 
    
    
    pix_points_cm = np.zeros((len(S),2))
    for i in range(len(S)):
        pix_points_cm[i,0] = round((pix_points[i,0] * w_rct) / image.shape[1])
        pix_points_cm[i,1] = round(( (image.shape[0] - pix_points[i,1]) * h_rct) / image.shape[0])
    
    
    # sensor_location=[]
    # for i in range(len(pix_points)):
    #         sensor_location = cv2.circle(image, (int(pix_points[i,0]),int(pix_points[i,1])), 4, (33,142,140), -1)         
    # plt.figure()
    # plt.imshow(sensor_location)
    # plt.title('Sensor locations')
    # plt.colorbar()
    # plt.show()
    # plt.savefig('Sensor locations.jpg',dpi=300)
    # sensor_location_rgb = cv2.cvtColor(sensor_location, cv2.COLOR_RGB2BGR)  # Convert to BGR for saving the image
    # path_circle = 'Data\Sensor locations.png'
    # cv2.imwrite(path_circle,sensor_location_rgb)
    
    return pix_points






def rec_around_sensor(image):
    
    length=5.4
    width=6.8
    h_rct = 180 
    w_rct = 226
    DeltaX = (w_rct/image.shape[1])  # m 
    DeltaY = (h_rct/image.shape[0])  # m
    increment_x = int(width/DeltaX)
    increment_y = int(length/DeltaY)
    
    return increment_x, increment_y
    
    
    
    

