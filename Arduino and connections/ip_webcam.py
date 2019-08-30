import urllib2
import cv2
import numpy as np
import sys
sys.path.insert(1, '/home/raloue/Desktop/technofarm_full_new/Plant_disease')
import plant_dis_2

plant_dis_2.load_model()

print("model has been loaded")

while True:
	url='http://192.168.43.1:8080/shot.jpg'
 	imgResponse = urllib2.urlopen(url)
 	imgNp = np.array(bytearray(imgResponse.read()),dtype=np.uint8)
 	img = cv2.imdecode(imgNp,-1)
 	plant_dis_2.callme(img)
 	if cv2.waitKey(1) & 0xFF == ord('q'):
 		break