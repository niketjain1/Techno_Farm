
import numpy as np
import pickle
import cv2
import os
from keras.preprocessing.image import img_to_array
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3' 


def load_model():
	global model1
	model1=pickle.load(open('/home/raloue/Desktop/technofarm_full_new/Plant_disease/cnn_model.pkl','r'))

def callme(img1):
	default_image_size = tuple((256, 256))
	def convert_image_to_array(image):
	    try:
	        if image is not None :
	            image = cv2.resize(image, default_image_size) 
	            return img_to_array(image)	
	        else :
				return np.array([])
	    except Exception as e:
	        return None

	l1=[]
	l1.append(convert_image_to_array(img1))
	x = np.array(l1, dtype=np.float16) / 255.0
	result=model1.predict(x,batch_size=None, verbose=0, steps=1)
	print(result)