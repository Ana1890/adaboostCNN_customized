import numpy as np
import tensorflow as tf
from tensorflow import keras
import imgaug.augmenters as iaa
from functions_generator import *

train_path = '//lxestudios/pacs/Eye/EyePacs/diabetic-retinopathy-detection/train/'


seq = iaa.Sequential([
    iaa.geometric.Rotate(rotate=(-20, 20), seed=None),
    iaa.flip.Fliplr(0.5, seed=None),
])

class DataGeneratorNext(keras.utils.Sequence):
    'Generates data for Keras'
    def __init__(self, images, labels, batch_size=2, dim=(224,224), n_channels=3,
                 n_classes=3, shuffle=True, augment=True, masked=True, proc=True):
        
        'Initialization'
        self.batch_size = batch_size
        self.labels = labels
        self.n_classes = n_classes
        self.images = images
        self.dim = dim
        self.n_channels = n_channels
        self.shuffle = shuffle
        self.augment = augment
        self.masked = masked
        self.proc = proc
        self.on_epoch_end()
        self.n = 0
        self.max = self.__len__()
                
        

    def __len__(self):
        'Denotes the number of batches per epoch'
        return int((len(self.images) / self.batch_size))

    def __getitem__(self, index):
        'Generate one batch of data'
        # Generate indexes of the batch
        indexes = self.indexes[index*self.batch_size:(index+1)*self.batch_size]
        # Find list of images and labels
            
        list_imgs_temp = [self.images[k] for k in indexes]
        list_labels_temp = [self.labels[k] for k in indexes]

        # Generate data
        X,y = self.__data_generation(list_imgs_temp, list_labels_temp)
        
        return X, y

    def on_epoch_end(self):
        'Updates indexes after each epoch'
        self.indexes = np.arange(len(self.images))
        if self.shuffle == True:
            np.random.shuffle(self.indexes)

    def __data_generation(self, list_imgs_temp, list_labels_temp):
        'Generates data containing batch_size samples' 
        
        # Initialization
        X = np.empty((self.batch_size, *self.dim, self.n_channels))
        y = np.empty((self.batch_size))
        #print(X.shape)
        # Generate data
        for i, img in enumerate(list_imgs_temp):

            X[i,] = load_image((train_path + img), self.masked, self.proc) # path

        for i, label in enumerate(list_labels_temp):
            # Store class
            y[i] = label
            
        if self.augment:
            X = seq(images=X)

        return X, y.astype('float64')
    
    def __next__(self):
        if self.n >= self.max:
            self.n = 0
        result = self.__getitem__(self.n)
        self.n += 1
        return result