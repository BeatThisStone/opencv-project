import cv2
#import time
import numpy as np
from time import sleep

#Definisco la soglia di pixel minima per considerare l'area rilevata come Vittima Gialla
light_min_pixels=10000
traffic_state: int = -1



cap = cv2.VideoCapture(0)
#cap.set(cv2.CAP_PROP_BUFFERSIZE, 2) # Più efficiente di 1 solo frame alla volta in memoria

# Generalmente la cattura avviene comunque a 640X480 anziché 1920X1080 che è il massimo per il modello di telecamera usato 
cap.set(3, 640)
cap.set(4, 480)

print("Test Riconoscimento Vittime Con Telecamera 1")
print("Premi 'q' Per Terminare")

while cv2.waitKey(1)!=ord('q'): # Premere q per terminare

    #InizioRilevamento: float = time.time()
    ret, frame = cap.read()
        
    hsv = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)

    lower_yellow = np.array([20,50,0])
    upper_yellow = np.array([40,255,255])           

    lower_red1 = np.array([0, 50,100])
    upper_red1= np.array([10,255,255])           
    lower_red2 = np.array([160, 50, 100])
    upper_red2 = np.array([180, 255, 255])
    
    
    lower_green = np.array([35, 40, 40])
    upper_green = np.array([85, 255, 255])

    
    mask_yellow = cv2.inRange(hsv, lower_yellow, upper_yellow)
    mask_red = cv2.bitwise_or(cv2.inRange(hsv, lower_red1, upper_red1),cv2.inRange(hsv, lower_red2, upper_red2))
    mask_green = cv2.inRange(hsv, lower_green, upper_green) 
    
    frame_yellow_filter = cv2.bitwise_and(frame,frame,mask=mask_yellow)
    frame_red_filter = cv2.bitwise_and(frame,frame,mask=mask_red)
    frame_green_filter = cv2.bitwise_and(frame,frame,mask=mask_green)

    yellow_pixel_count = cv2.countNonZero(mask_yellow)
    red_pixel_count = cv2.countNonZero(mask_red)    
    green_pixel_count = cv2.countNonZero(mask_green)    

    traffic_frame = cv2.add(frame_yellow_filter, cv2.add(frame_red_filter, frame_green_filter))
    cv2.imshow('Semaforo', traffic_frame)

    #TempoRiconoscimento=time.time()-InizioRilevamento
    sleep(0.1)
    best_color_value = max(yellow_pixel_count, red_pixel_count, green_pixel_count)
    new_state: int = 0
    if best_color_value < light_min_pixels:
        pass
    elif best_color_value == red_pixel_count:
        new_state = 1
    elif best_color_value == yellow_pixel_count:
        new_state = 2
    else:
        new_state = 3
        
    if new_state != traffic_state:
        traffic_state = new_state
        match traffic_state:
            case 0:
                print("Semaforo non in vista")
            case 1:
                print("STOP")
            case 2:
                print("RALLENTA")
            case 3:
                print("PROSEGUI")
    
# Rilascia gli oggetti e termina le operazioni video
cap.release()
cv2.destroyAllWindows()
    
