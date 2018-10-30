import keras
import os
import cv2
import numpy as np
from keras.models import load_model

model = load_model('netG.h5')

test_folder = 'pix2pix/testA'
candidate_folder = 'pix2pix/candidateA'

os.mkdir('pix2pix/test_predicted_pgan')
os.mkdir('pix2pix/candidate_predicted')

list_all_test_dir = os.listdir(test_folder)
list_all_candidate_dir = os.listdir(candidate_folder)

img_list_test = [j for j in list_all_test_dir if j.endswith('.tif')]
img_list_candidate = [h for h in list_all_candidate_dir if h.endswith('.tif')]
    
for idx in range(len(img_list_test)):
    test_name = img_list_test[idx]
    test_path = test_folder+'/'+test_name
    img_test = cv2.imread(test_path,-1)    
    img_test = img_test/750.0-1.0
    img_test = img_test.reshape(1,256,256,1)    
    img_test_predicted = model.predict(img_test)
    img_test_predicted = img_test_predicted.reshape(256,256)
    img_test_predicted = ((img_test_predicted+1.0)*750.0).astype('uint16')
    cv2.imwrite('pix2pix/test_predicted_pgan/'+test_name,img_test_predicted)

for idx in range(len(img_list_candidate)):
    candidate_name = img_list_candidate[idx]
    candidate_path = candidate_folder+'/'+candidate_name
    img_candi = cv2.imread(candidate_path,-1)    
    img_candi = img_candi/750.0-1.0
    img_candi = img_candi.reshape(1,256,256,1)    
    img_candi_predicted = model.predict(img_candi)
    img_candi_predicted = img_candi_predicted.reshape(256,256)
    img_candi_predicted = ((img_candi_predicted+1.0)*750.0).astype('uint16')
    cv2.imwrite('pix2pix/candidate_predicted/'+candidate_name,img_candi_predicted)
