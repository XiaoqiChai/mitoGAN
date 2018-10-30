#%% setting
import os
os.environ['KERAS_BACKEND']='tensorflow' # can choose theano, tensorflow, cntk
os.environ['THEANO_FLAGS']='floatX=float32,device=cuda,optimizer=fast_run,dnn.library_path=/usr/lib'
#os.environ['THEANO_FLAGS']='floatX=float32,device=cuda,optimizer=fast_compile,dnn.library_path=/usr/lib'
import keras.backend as K
if os.environ['KERAS_BACKEND'] =='theano':
    channel_axis=1
    K.set_image_data_format('channels_first')
    channel_first = True
else:
    K.set_image_data_format('channels_last')
    channel_axis=-1
    channel_first = False

#%% imports
from keras.models import Model
from keras.layers import Conv2D, ZeroPadding2D, BatchNormalization, Input, Dropout
from keras.layers import Conv2DTranspose, Activation, Cropping2D
from keras.layers import Concatenate
from keras.layers.advanced_activations import LeakyReLU
from keras.initializers import RandomNormal
from keras.optimizers import Adam
import numpy as np
import glob
import cv2
from random import randint, shuffle
import time
import matplotlib.pyplot as plt

#%% intializer
def __conv_init(a):
    print("conv_init", a)
    k = RandomNormal(0, 0.02)(a) # for convolution kernel
    k.conv_weight = True    
    return k

#%% DCGAN
def conv2d(f, *a, **k):
    return Conv2D(f, kernel_initializer = RandomNormal(0, 0.02), *a, **k)
def batchnorm():
    return BatchNormalization(momentum=0.9, axis=channel_axis, epsilon=1.01e-5,
                                   gamma_initializer = RandomNormal(1., 0.02))
def BASIC_D(nc_in, nc_out, ndf, max_layers=3):
    """DCGAN_D(nc, ndf, max_layers=3)
       nc: channels
       ndf: filters of the first layer
       max_layers: max hidden layers
    """    
    if channel_first:
        input_a, input_b =  Input(shape=(nc_in, None, None)), Input(shape=(nc_out, None, None))
    else:
        input_a, input_b = Input(shape=(None, None, nc_in)), Input(shape=(None, None, nc_out))
    _ = Concatenate(axis=channel_axis)([input_a, input_b])
    _ = conv2d(ndf, kernel_size=4, strides=2, padding="same", name = 'First') (_)
    _ = LeakyReLU(alpha=0.2)(_)
    
    for layer in range(1, max_layers):        
        out_feat = ndf * min(2**layer, 8)
        _ = conv2d(out_feat, kernel_size=4, strides=2, padding="same", 
                   use_bias=False, name = 'pyramid.{0}'.format(layer)             
                        ) (_)
        _ = batchnorm()(_, training=1)        
        _ = LeakyReLU(alpha=0.2)(_)
    
    out_feat = ndf*min(2**max_layers, 8)
    _ = ZeroPadding2D(1)(_)
    _ = conv2d(out_feat, kernel_size=4,  use_bias=False, name = 'pyramid_last') (_)
    _ = batchnorm()(_, training=1)
    _ = LeakyReLU(alpha=0.2)(_)
    
    # final layer
    _ = ZeroPadding2D(1)(_)
    _ = conv2d(1, kernel_size=4, name = 'final'.format(out_feat, 1), 
               activation = "sigmoid") (_)    
    return Model(inputs=[input_a, input_b], outputs=_)


def UNET_G(isize, nc_in=3, nc_out=3, ngf=64, fixed_input_size=True):    
    max_nf = 8*ngf    
    def block(x, s, nf_in, use_batchnorm=True, nf_out=None, nf_next=None):
        # print("block",x,s,nf_in, use_batchnorm, nf_out, nf_next)
        assert s>=2 and s%2==0
        if nf_next is None:
            nf_next = min(nf_in*2, max_nf)
        if nf_out is None:
            nf_out = nf_in
        x = conv2d(nf_next, kernel_size=4, strides=2, use_bias=(not (use_batchnorm and s>2)),
                   padding="same", name = 'conv_{0}'.format(s)) (x)
        if s>2:
            if use_batchnorm:
                x = batchnorm()(x, training=1)
            x2 = LeakyReLU(alpha=0.2)(x)
            x2 = block(x2, s//2, nf_next)
            x = Concatenate(axis=channel_axis)([x, x2])            
        x = Activation("relu")(x)
        x = Conv2DTranspose(nf_out, kernel_size=4, strides=2, use_bias=not use_batchnorm,
                            kernel_initializer = RandomNormal(0, 0.02),          
                            name = 'convt.{0}'.format(s))(x)        
        x = Cropping2D(1)(x)
        if use_batchnorm:
            x = batchnorm()(x, training=1)
        if s <=8:
            x = Dropout(0.5)(x, training=1)
        return x
    
    s = isize if fixed_input_size else None
    if channel_first:
        _ = inputs = Input(shape=(nc_in, s, s))
    else:
        _ = inputs = Input(shape=(s, s, nc_in))        
    _ = block(_, isize, nc_in, False, nf_out=nc_out, nf_next=ngf)
    _ = Activation('tanh')(_)
    return Model(inputs=inputs, outputs=[_])

#%% data
def load_data(file_pattern):
    return glob.glob(file_pattern)
def read_image(fn, direction=0):
    im = cv2.imread(fn,-1)
    arr = np.array(im)/750.0-1.0
    arr = np.reshape(arr,(loadSize, loadSize*2,1))
    w1,w2 = (loadSize-imageSize)//2,(loadSize+imageSize)//2
    h1,h2 = w1,w2
    imgA = arr[h1:h2, loadSize+w1:loadSize+w2, :]
    imgB = arr[h1:h2, w1:w2, :]
    if randint(0,1):
        imgA=imgA[:,::-1]
        imgB=imgB[:,::-1]
    if channel_first:
        imgA = np.moveaxis(imgA, 2, 0)
        imgB = np.moveaxis(imgB, 2, 0)
    if direction==0:
        return imgA, imgB
    else:
        return imgB,imgA
    
def minibatch(dataAB, batchsize, direction=0):
    length = len(dataAB)
    i = 0   
    while i+batchsize < length:
        dataA = []
        dataB = []
        for j in range(i,i+batchsize):
            imgA,imgB = read_image(dataAB[j], direction)
            dataA.append(imgA)
            dataB.append(imgB)
        dataA = np.float32(dataA)
        dataB = np.float32(dataB)
        i+=batchsize
        yield dataA, dataB  
        
def showX(X, rows=1, save_file=None):
    assert X.shape[0]%rows == 0
    int_X = ( (X+1)/2*255).clip(0,255).astype('uint8')
    if channel_first:
        int_X = np.moveaxis(int_X.reshape(-1,3,imageSize,imageSize), 1, 3)
    else:
        int_X = int_X.reshape(-1,imageSize,imageSize, 1)
    int_X = int_X.reshape(rows, -1, imageSize, imageSize,1).swapaxes(1,2).reshape(rows*imageSize,-1)
#    display(Image.fromarray(int_X,"L"))
    if save_file:
        plt.imshow(int_X, cmap='gray')
        plt.axis("off")
        plt.savefig(save_file)
    else:
        plt.imshow(int_X, cmap='gray')
        plt.axis("off")

#%% build DCGAN    
    
nc_in = 1    # number of input channels
nc_out = 1   # number of output channels
ngf = 64     # number of filters of the first layer in generator
ndf = 64     # number of filters of the first layer in discriminator

loadSize = 256
imageSize = 256
lrD = 2e-4
lrG = 2e-4

netD = BASIC_D(nc_in, nc_out, ndf)
netD.summary()

netG = UNET_G(imageSize, nc_in, nc_out, ngf)
netG.summary()

real_A = netG.input
fake_B = netG.output
netG_generate = K.function([real_A], [fake_B])
real_B = netD.inputs[1]
output_D_real = netD([real_A, real_B])
output_D_fake = netD([real_A, fake_B])

loss_fn = lambda output, target : -K.mean(K.log(output+1e-12)*target+K.log(1-output+1e-12)*(1-target))

loss_D_real = loss_fn(output_D_real, K.ones_like(output_D_real))
loss_D_fake = loss_fn(output_D_fake, K.zeros_like(output_D_fake))
loss_G_fake = loss_fn(output_D_fake, K.ones_like(output_D_fake))

loss_L1 = K.mean(K.abs(fake_B-real_B))

loss_D = loss_D_real +loss_D_fake
training_updates = Adam(lr=lrD, beta_1=0.5).get_updates(netD.trainable_weights,[],loss_D)
netD_train = K.function([real_A, real_B],[loss_D/2], training_updates)

loss_G = loss_G_fake   + 100 * loss_L1
training_updates = Adam(lr=lrG, beta_1=0.5).get_updates(netG.trainable_weights,[], loss_G)
netG_train = K.function([real_A, real_B], [loss_G_fake, loss_L1], training_updates)


#%% training
data = "mito"
direction = 0
batchSize = 5
epochs = 800
display_iters = 100
save_path = 'temp_results'
os.mkdir(save_path)

trainAB = load_data('pix2pix/train/*.tif')
valAB = load_data('pix2pix/val/*.tif')
assert len(trainAB) and len(valAB)

errL1 = errG = errD = 0
errD_all = []
errG_all = []
errL1_all = []
val_batch = minibatch(valAB, batchSize, direction)
for ep in range(epochs): 
    t0 = time.time()
    shuffle(trainAB)
    train_batch = minibatch(trainAB, batchSize, direction)
    for trainA, trainB in train_batch:
#        trainA, trainB = next(train_batch)
        errD += netD_train([trainA, trainB])[0]
        errG_, errL1_ = netG_train([trainA, trainB])
        errG += errG_
        errL1 += errL1_
    print('[%d/%d] Loss_D: %f Loss_G: %f loss_L1: %f'
    % (ep, epochs, errD, errG, errL1), time.time()-t0)
    errD_all.append(errD)
    errG_all.append(errG)
    errL1_all.append(errL1)
    errL1 = errG = errD = 0
    if ep % display_iters==0:
        try: 
            valA, valB = next(val_batch)
        except:
            val_batch = minibatch(valAB, batchSize, direction)
            valA, valB = next(val_batch)
        fakeB, = netG_generate([valA])
        # save_file = None
        save_file = os.path.join(save_path,'val_epoch%d.png' % ep) 
        showX(np.concatenate([valA, valB, fakeB], axis=0), 3, save_file)
        netG.save(os.path.join(save_path,'netG_epoch%d.h5' % ep))
        netD.save(os.path.join(save_path,'netD_epoch%d.h5' % ep))

netG.save('netG.h5')
netD.save('netD.h5')

#%%

plt.figure(1)
plt.plot(range(epochs),errG_all)
plt.title("errG")
plt.savefig(os.path.join(save_path,"errG.png"))

plt.figure(2)
plt.plot(range(epochs),errD_all)
plt.title("errD")
plt.savefig(os.path.join(save_path,"errD.png"))

plt.figure(3)
plt.plot(range(epochs),errL1_all)
plt.title("errL1")
plt.savefig(os.path.join(save_path,"errL1.png"))
