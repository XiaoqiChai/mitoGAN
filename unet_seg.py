import numpy as np
import scipy.io as io
from keras.models import Model
from keras.layers import Input, Conv2D, MaxPooling2D, UpSampling2D, Dropout, Cropping2D
from keras.layers import merge 
from keras.layers import concatenate
from keras.optimizers import SGD
from keras.callbacks import ModelCheckpoint, LearningRateScheduler
import matplotlib.pyplot as plt

class myUnet(object):

    def __init__(self, img_rows = 200, img_cols = 200):

        self.img_rows = img_rows
        self.img_cols = img_cols

    def get_unet(self):

        inputs = Input((self.img_rows, self.img_cols,1))
        # print inputs.shape

        conv1 = Conv2D(64, kernel_size=(3,3), activation = 'relu', padding = 'same', kernel_initializer = 'he_normal')(inputs)
        # print "conv1 shape:",conv1.shape 
        conv1 = Conv2D(64, 3, activation = 'relu', padding = 'same', kernel_initializer = 'he_normal')(conv1)
        # print("conv1 shape:",conv1.shape)
        pool1 = MaxPooling2D(pool_size=(2, 2))(conv1)
        # print "pool1 shape:",pool1.shape    

        conv2 = Conv2D(128, 3, activation = 'relu', padding = 'same', kernel_initializer = 'he_normal')(pool1)
        # print "conv2 shape:",conv2.shape
        conv2 = Conv2D(128, 3, activation = 'relu', padding = 'same', kernel_initializer = 'he_normal')(conv2)
        # print("conv2 shape:",conv2.shape)    
        pool2 = MaxPooling2D(pool_size=(2, 2))(conv2)
        # print "pool2 shape:",pool2.shape    

        conv3 = Conv2D(256, 3, activation = 'relu', padding = 'same', kernel_initializer = 'he_normal')(pool2)
        # print "conv3 shape:",conv3.shape
        conv3 = Conv2D(256, 3, activation = 'relu', padding = 'same', kernel_initializer = 'he_normal')(conv3)
        # print("conv3 shape: ", conv3.shape)
        drop3 = Dropout(0.5)(conv3)
        pool3 = MaxPooling2D(pool_size=(2, 2))(conv3)
        # print("pool3 shape:",pool3.shape)    

        conv4 = Conv2D(512, 3, activation = 'relu', padding = 'same', kernel_initializer = 'he_normal')(pool3)
        conv4 = Conv2D(512, 3, activation = 'relu', padding = 'same', kernel_initializer = 'he_normal')(conv4)
        drop4 = Dropout(0.5)(conv4)
        # print("drop4.shape: ", drop4.shape)
        
        # pool4 = MaxPooling2D(pool_size=(2, 2))(drop4)

        # conv5 = Conv2D(1024, 3, activation = 'relu', padding = 'same', kernel_initializer = 'he_normal')(pool4)
        # conv5 = Conv2D(1024, 3, activation = 'relu', padding = 'same', kernel_initializer = 'he_normal')(conv5)
        # drop5 = Dropout(0.5)(conv5)

        # up6 = Conv2D(512, 2, activation = 'relu', padding = 'same', kernel_initializer = 'he_normal')(UpSampling2D(size = (2,2))(drop5))
        # print "up6.shape: ", up6.shape    #(?, 64, 64, 512)
        # merge6 = merge([drop4,up6], mode = 'concat', concat_axis = 3)   
        # merge6 = concatenate([drop4,up6], axis = 3)    
        # print "concatenate6 done. shape: ", merge6.shape    #(?, 64, 64, 1024)
        # conv6 = Conv2D(512, 3, activation = 'relu', padding = 'same', kernel_initializer = 'he_normal')(merge6)
        # conv6 = Conv2D(512, 3, activation = 'relu', padding = 'same', kernel_initializer = 'he_normal')(conv6)
        # print "conv6 shape: ", conv6.shpae   #(?, 64, 64, 512)

        up7 = Conv2D(256, 2, activation = 'relu', padding = 'same', kernel_initializer = 'he_normal')(UpSampling2D(size = (2,2))(drop4))
        # print("up7 shape: ", up7.shape)
        merge7 = concatenate([conv3,up7], axis = 3)
        # print("merge7 shape: ", merge7.shape)
        conv7 = Conv2D(256, 3, activation = 'relu', padding = 'same', kernel_initializer = 'he_normal')(merge7)
        conv7 = Conv2D(256, 3, activation = 'relu', padding = 'same', kernel_initializer = 'he_normal')(conv7)
        # print("conv7 shpae: ", conv7.shape)

        up8 = Conv2D(128, 2, activation = 'relu', padding = 'same', kernel_initializer = 'he_normal')(UpSampling2D(size = (2,2))(conv7))
        # print "up8 shape: ", up8.shape  
        merge8 = concatenate([conv2,up8], axis = 3)
        # print("merge8 shape: ", merge8.shape)
        conv8 = Conv2D(128, 3, activation = 'relu', padding = 'same', kernel_initializer = 'he_normal')(merge8)
        conv8 = Conv2D(128, 3, activation = 'relu', padding = 'same', kernel_initializer = 'he_normal')(conv8)

        up9 = Conv2D(64, 2, activation = 'relu', padding = 'same', kernel_initializer = 'he_normal')(UpSampling2D(size = (2,2))(conv8))
        # print "up9 shape: ", up9.shape  
        merge9 = concatenate([conv1,up9], axis = 3)
        # print("merge9 shape: ", merge9.shape)
        conv9 = Conv2D(64, 3, activation = 'relu', padding = 'same', kernel_initializer = 'he_normal')(merge9)
        conv9 = Conv2D(64, 3, activation = 'relu', padding = 'same', kernel_initializer = 'he_normal')(conv9)
        # conv9 = Conv2D(2, 3, activation = 'relu', padding = 'same', kernel_initializer = 'he_normal')(conv9)
        conv9 = Conv2D(2, 1, activation = 'relu', padding = 'same', kernel_initializer = 'he_normal')(conv9)
        # print("conv9 shape:",conv9.shape)
        
        conv10 = Conv2D(1, 1, activation = 'sigmoid')(conv9)
        # print("conv10 shape:",conv10.shape)  
        
        
        # model = Model(input = inputs, output = conv10)
        model = Model(inputs = inputs, outputs = conv10) 
        return model

#%% load data
data_source = 'sim_data' # 'cycle_data'  'real_data' 'pix2pix_data' 'sim_data'
file_suffix = 'sim_part2'  # 'p2p_ep800' 'part2_p2p_ep1000' 'part2' 'cycle_ep200'
# ============================= train data ===============================
imgs_train = np.load('data/'+ data_source + '/npydata/images_train_' + file_suffix + '.npy').astype('float32')
img_max_thr = imgs_train.max()
imgs_train = imgs_train/img_max_thr
#imgs_train = imgs_train/750.0-1

masks_train = np.load('data/cycle_data/npydata/masks_train_cycle_ep200.npy').astype('float32')
masks_train[masks_train>0]=1
masks_train[masks_train<=0]=0

# ============================= val data ===============================
imgs_val = np.load('data/real_data/npydata/images_val_part2.npy').astype('float32')
imgs_val = imgs_val/img_max_thr
#imgs_val = imgs_val/750.0-1
masks_val = np.load('data/real_data/npydata/masks_val_part2.npy').astype('float32')
masks_val[masks_val>0]=1
masks_val[masks_val<=0]=0

# ============================= test data ===============================
imgs_test = np.load('data/real_data/npydata/images_test.npy').astype('float32')
imgs_test = imgs_test/img_max_thr
#imgs_test = imgs_test/750.0-1
masks_test = np.load('data/real_data/npydata/masks_test.npy').astype('float32')
masks_test[masks_test>0]=1
masks_test[masks_test<=0]=0

# imgs_train, masks_train, imgs_val, masks_val, imgs_test, masks_test     


#%% training
bs = 4
epochs = 50
result_path = 'test_results/sim_seg' # pix2pix_seg, real_seg, cycle_seg, sim_seg
run_no = 'run2'

model = myUnet().get_unet()        
model.compile(optimizer = SGD(lr = 0.01), loss = 'binary_crossentropy', metrics = ['accuracy'])
#model_checkpoint = ModelCheckpoint(result_path+'/'+file_suffix+'_model_'+run_no+'.h5', monitor='val_loss',verbose=1, save_best_only=True)
#print('Fitting model...')
history = model.fit(imgs_train, masks_train, validation_data=(imgs_val,masks_val), batch_size=bs, epochs=epochs, verbose=1, shuffle=True, callbacks=[model_checkpoint])
#history = model.fit(imgs_train, masks_train, batch_size=bs, epochs=epochs, verbose=1, shuffle=True, callbacks=[model_checkpoint])


#%%
plt.plot(history.history['loss'])
plt.plot(history.history['val_loss'])
plt.legend(['train_loss','val_loss'])
plt.title('loss')
#plt.show()
plt.savefig(result_path+'/'+file_suffix+'_loss_'+run_no+'.png')

plt.figure()
plt.plot(history.history['acc'])
plt.plot(history.history['val_acc'])
plt.legend(['train_acc','val_acc'])
plt.title('acc')
#plt.show()
plt.savefig(result_path+'/'+file_suffix+'_acc_'+run_no+'.png')

#%% testing
model.load_weights(result_path+'/'+file_suffix+'_model_'+run_no+'.h5')

test_loss, test_acc = model.evaluate(imgs_test, masks_test, batch_size=bs, verbose=1)
print("test_loss: ",test_loss)
print("test_acc: ",test_acc)

print('predict test data')
test_predict = model.predict(imgs_test, batch_size=bs, verbose=1)
##np.save()
#io.savemat(result_path+'/'+file_suffix+'_test_prediction_'+run_no+'.mat', {'prediction':test_predict})
#print('prediction saved')

thr = 0.5
test_predict[test_predict>=thr]=1
test_predict[test_predict<thr]=0
tmp = np.add(masks_test, test_predict).reshape(masks_test.shape[0],-1)
intersection = np.sum(tmp==2,axis=1)
union = np.sum(tmp>=1, axis=1)
iou = np.divide(intersection, union)

for i in range(iou.shape[0]):
    print(iou[i])
print('intersect_over_union (iou) acc: ', iou.mean()) 
