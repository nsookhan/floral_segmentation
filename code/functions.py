########################################################################################################################
########################################################################################################################
################################################################################################################
### load packages ###
################################################################################################################
########################################################################################################################
########################################################################################################################
import os

import segmentation_models.utils

os.environ['TF_GPU_ALLOCATOR'] = 'cuda_malloc_async'
os.environ["SM_FRAMEWORK"] = "tf.keras"
import csv
import gc
import random
from pathlib import Path
from shutil import copy
from re import findall
from functools import reduce
from itertools import compress
from matplotlib import pyplot as plt

import segmentation_models as sm
import numpy as np
import tensorflow as tf
import tensorflow.keras.utils as ku
import keras
import cv2

from osgeo import gdal
from osgeo import ogr
from osgeo import gdalconst

from sklearn.model_selection import train_test_split

from keras.models import load_model
from sklearn.preprocessing import MinMaxScaler
from sklearn import metrics
from keras import layers as kl

from sklearn.preprocessing import MinMaxScaler
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras.callbacks import EarlyStopping
from sklearn.utils import shuffle
from imgaug import augmenters as iaa
from imgaug.augmentables.segmaps import SegmentationMapsOnImage
from imgaug.augmenters.meta import Sometimes

from segmentation_models.utils import set_regularization

from numpy.lib.stride_tricks import as_strided
import tensorflow_addons as tfa
import patchify
import time
########################################################################################################################
########################################################################################################################
################################################################################################################
### PREPROCESS_DATA functions ###
################################################################################################################
########################################################################################################################
########################################################################################################################

########################################################################################################################
### make directory ###
########################################################################################################################
def mk_dir(path):
    try: os.mkdir(path)
    except OSError as error:
        print(error)

########################################################################################################################
### create site level directory with sub-directories ###
########################################################################################################################
def site_dir(site_name=None,tile_size=None):
    tile_name = "tile_" + str(tile_size)
    path = 'data\\' + site_name + '\\' + tile_name
    # create directories
    # base
    mk_dir('data\\' + site_name)
    mk_dir(path)
    # img
    mk_dir(path + '\\img')
    mk_dir(path + '\\img\\tile')
    mk_dir(path + '\\img\\full')
    # msk
    mk_dir(path + '\\msk')
    mk_dir(path + '\\msk\\tile')
    mk_dir(path + '\\msk\\full')
    # grd
    mk_dir(path + '\\grd')
    mk_dir(path + '\\grd\\tile')
    mk_dir(path + '\\grd\\full')

    print(site_name + ' directory created')

def trainvaltest_dir(tile_size=None):
    mk_dir('data\\' + ('tile_' + str(tile_size)))
    mk_dir('data\\' + ('tile_' + str(tile_size)) + '\\train_img')
    mk_dir('data\\' + ('tile_' + str(tile_size)) + '\\train_img\\train')
    mk_dir('data\\' + ('tile_' + str(tile_size)) + '\\val_img')
    mk_dir('data\\' + ('tile_' + str(tile_size)) + '\\val_img\\val')
    mk_dir('data\\' + ('tile_' + str(tile_size)) + '\\test_img')
    mk_dir('data\\' + ('tile_' + str(tile_size)) + '\\test_img\\test')
    mk_dir('data\\' + ('tile_' + str(tile_size)) + '\\train_msk')
    mk_dir('data\\' + ('tile_' + str(tile_size)) + '\\train_msk\\train')
    mk_dir('data\\' + ('tile_' + str(tile_size)) + '\\val_msk')
    mk_dir('data\\' + ('tile_' + str(tile_size)) + '\\val_msk\\val')
    mk_dir('data\\' + ('tile_' + str(tile_size)) + '\\test_msk')
    mk_dir('data\\' + ('tile_' + str(tile_size)) + '\\test_msk\\test')
    mk_dir('data\\' + ('tile_' + str(tile_size)) + '\\metrics')
    mk_dir('data\\' + ('tile_' + str(tile_size)) + '\\model')
    mk_dir('data\\' + ('tile_' + str(tile_size)) + '\\run')
    mk_dir('data\\' + ('tile_' + str(tile_size)) + '\\train2_img')
    mk_dir('data\\' + ('tile_' + str(tile_size)) + '\\train2_img\\train')
    mk_dir('data\\' + ('tile_' + str(tile_size)) + '\\val2_img')
    mk_dir('data\\' + ('tile_' + str(tile_size)) + '\\val2_img\\val')
    mk_dir('data\\' + ('tile_' + str(tile_size)) + '\\train2_msk')
    mk_dir('data\\' + ('tile_' + str(tile_size)) + '\\train2_msk\\train')
    mk_dir('data\\' + ('tile_' + str(tile_size)) + '\\val2_msk')
    mk_dir('data\\' + ('tile_' + str(tile_size)) + '\\val2_msk\\val')
########################################################################################################################
### drop excess pixels ###
########################################################################################################################
def process_raster(img_path=None,site_name=None, tile_size=None, folder_name=None):
    tile_name = "tile_" + str(tile_size)
    # open raster
    img = gdal.Open(img_path)
    # to array
    img_1 = np.array(img.GetRasterBand(1).ReadAsArray())
    # get height and width divisible by tile_size
    img_height, img_width = img_1.shape
    h = (img_height // tile_size) * tile_size
    w = (img_width // tile_size) * tile_size
    # drop pixels
    x = gdal.Translate('data\\' + site_name + '\\' + tile_name + '\\' + folder_name + '\\full\\' + folder_name + '.tif',
                       img,
                       srcWin=[(img_width - w), (img_height - h), w, h],
                       creationOptions='TFW=yes')
    x = None
    img = None
    print(site_name + ' ' + folder_name + ' clipped')

########################################################################################################################
### tile data ###
########################################################################################################################
def tile_raster(site_name=None, tile_size=None, folder_name=None, nchannels=None):
    tile_name = "tile_" + str(tile_size)
    img = gdal.Open('data\\' + site_name + '\\' + tile_name + '\\' + folder_name + '\\full\\' + folder_name + '.tif')
    # remove projection
    img = [img.GetRasterBand(i).ReadAsArray() for i in range(1,nchannels+1)]
    img = np.stack(img, axis=2)
    # to 8bit
    img = img.astype(np.uint8)
    # set parameters
    img_height, img_width, img_channels = img.shape
    number_tiles_h = img_height // tile_size
    number_tiles_w = img_width // tile_size
    # tile data
    tile_img = img.reshape((number_tiles_h, tile_size,
                            number_tiles_w, tile_size,
                            img_channels)).swapaxes(1, 2)
    # flatten array
    tile_img = tile_img.reshape(-1, tile_size, tile_size, img_channels)
    # reconstruct spatial relationships
    # tile_img = tile_img.reshape(number_tiles_h, number_tiles_w, tile_size, tile_size, img_channels)
    # write to disk
    for i in range(tile_img.shape[0]):
        cv2.imwrite('data\\' + site_name + '\\' + tile_name + '\\' + folder_name + '\\tile\\' + site_name + '_' + str(i) + '.tif',
                    tile_img[i, :, :, ::-1])
    print(site_name + ' ' + folder_name + ' tiled')

def tile_raster2(full_path=None, tile_path=None, site_name=None, tile_size=None, nchannels=None):
    tile_name = "tile_" + str(tile_size)
    img = gdal.Open(full_path)
    # remove projection
    img = [img.GetRasterBand(i).ReadAsArray() for i in range(1,nchannels+1)]
    img = np.stack(img, axis=2)
    # to 8bit
    img = img.astype(np.uint8)
    # set parameters
    img_height, img_width, img_channels = img.shape
    number_tiles_h = img_height // tile_size
    number_tiles_w = img_width // tile_size
    # tile data
    tile_img = img.reshape((number_tiles_h, tile_size,
                            number_tiles_w, tile_size,
                            img_channels)).swapaxes(1, 2)
    # flatten array
    tile_img = tile_img.reshape(-1, tile_size, tile_size, img_channels)
    # reconstruct spatial relationships
    # tile_img = tile_img.reshape(number_tiles_h, number_tiles_w, tile_size, tile_size, img_channels)
    # write to disk
    for i in range(tile_img.shape[0]):
        cv2.imwrite(tile_path + site_name + '_' + str(i) + '.tif',
                    tile_img[i, :, :, ::-1])
    print(full_path + ' tiled')

########################################################################################################################
### rasterize manual masking layer ###
########################################################################################################################
def rasterize(site_name=None, tile_size=None, folder_name=None, msk_path=None, img_path=None, attr=None):
    tile_name = "tile_" + str(tile_size)
    ### input data ###
    msk = ogr.Open(msk_path)
    msk_l = msk.GetLayer()
    img = gdal.Open(img_path, gdalconst.GA_ReadOnly)
    ### rasterize ###
    # settings
    target_ds = gdal.GetDriverByName('GTiff').Create('data\\' + site_name + '\\' + tile_name + '\\' + folder_name + '\\full\\' + folder_name + '.tif',
                                                     img.RasterXSize, img.RasterYSize, 1, gdal.GDT_Byte)
    target_ds.SetGeoTransform(img.GetGeoTransform())
    target_ds.SetProjection(img.GetProjection())
    band = target_ds.GetRasterBand(1)
    band.SetNoDataValue(0)
    band.FlushCache()
    # rasterize
    x = gdal.RasterizeLayer(target_ds, [1], msk_l, options=['ATTRIBUTE=' + attr])
    target_ds = None
    x = None
    print(site_name + ' ' + folder_name + ' rasterized')

########################################################################################################################
### CREATE LIST OF MANUALLY MASKED TILES ###
########################################################################################################################
def create_tile_list(site_name=None,tile_size=None,folder_name=None,project_name=None):
    tile_name = "tile_" + str(tile_size)
    path = 'data\\' + site_name + '\\' + tile_name + '\\' + folder_name + '\\tile\\'
    # get tile names
    tile_list_names = os.listdir(path)
    # sort names by grid index
    index = np.argsort([int(findall('(?<=_)[0-9]+(?=\\.tif)', i)[0]) for i in tile_list_names])
    tile_list_names = [tile_list_names[i] for i in index]
    ### LOAD IMAGES ###
    tile_list = [cv2.imread(path + i, 0) for i in tile_list_names]
    tile_list = np.array(tile_list)
    ### CREATE LIST OF MANUALLY MASKED TILES ###
    check = [(tile_list[i, :, :] > 0).sum() / (tile_list[i, :, :] > -1).sum() for i in range(tile_list.shape[0])]
    check = np.array(check)
    check = (check > 0.95).nonzero()[0]
    # to tile names
    check = np.char.mod("%d", check)
    check = np.char.add('_', check)
    check = np.char.add(site_name, check)
    check = np.char.add(check, '.tif')
    ### WRITE TO DISK ###
    tile_list_path = '..\\' + project_name + '\\data\\' + site_name + '\\' + tile_name + '\\' + folder_name + '\\' + 'tile_list.csv'
    #path = Path(__file__).parent / tile_list_path
    path = tile_list_path
    with open(path, 'a', newline='') as f:
        write = csv.writer(f)
        write.writerow(check)
    print(site_name + ' tile list created')

########################################################################################################################
### load tile list ###
########################################################################################################################
def load_tile_list(site_name=None,tile_size=None,folder_name=None,project_name=None):
    tile_name = "tile_" + str(tile_size)
    tile_path = '..\\' + project_name + '\\data\\' + site_name + '\\' + tile_name + '\\' + folder_name + '\\' + 'tile_list.csv'
    # get masked tile IDs
    tile_list = list()
    # path = Path(__file__).parent / tile_path
    path = tile_path
    with open(path, 'r', newline='') as f:
        reader = csv.reader(f)
        for row in reader:
            tile_list = row
    # return
    return tile_list

########################################################################################################################
### load/write tile ###
########################################################################################################################
def load_tile(path=None,nchannels=None):
    if nchannels > 1:
        y = cv2.imread(path, 1)[:, :, ::-1]
    else:
        y = cv2.imread(path, 0)
    return y

def write_tile(path=None,array=None,nchannels=None):
    if nchannels > 1:
        cv2.imwrite(path, array[:, :, ::-1])
    else:
        cv2.imwrite(path, array)


########################################################################################################################
### load tiles in tile list ###
########################################################################################################################
# def get_tile_from_list(site_name=None,tile_size=None,folder_name=None,nchannel=None,tile_list=None):
#     tile_name = "tile_" + str(tile_size)
#     path = 'data\\' + site_name + '\\' + tile_name + '\\' + folder_name + '\\tile\\'
#     # iterate across tile list and load tiles
#     if nchannel>1:
#         flag = 1
#         raster_list = [cv2.imread(path + i, flag)[:, :, ::-1] for i in tile_list]
#     else:
#         flag = 0
#         raster_list = [cv2.imread(path + i, cv2.IMREAD_UNCHANGED).astype(np.uint8) for i in tile_list]
#     raster_list = np.array(raster_list)
#     # return tiles
#     return raster_list
def get_tile_from_list(site_name=None,tile_size=None,folder_name=None,nchannels=None,tile_list=None):
    tile_name = "tile_" + str(tile_size)
    path = 'data\\' + site_name + '\\' + tile_name + '\\' + folder_name + '\\tile\\'
    # iterate across tile list and load tiles
    raster_list = [load_tile(path + i, nchannels) for i in tile_list]
    raster_list = np.array(raster_list)
    # return tiles
    return raster_list

def get_tile_from_list2(path=None,nchannels=None,tile_list=None):
    # iterate across tile list and load tiles
    raster_list = [load_tile(path + i, nchannels) for i in tile_list]
    raster_list = np.array(raster_list)
    # return tiles
    return raster_list

########################################################################################################################
### train, validation, test split ###
########################################################################################################################
def train_val_test_split(site_name=None,tile_size=None, tile_list=None,
                         test_prop=0.2,val_prop=0.25,
                         seed=None,
                         path_train_img=None, path_train_msk=None,
                         path_val_img=None, path_val_msk=None,
                         path_test_img=None, path_test_msk=None):
    # get path
    tile_name = 'tile_' + str(tile_size)
    path_src = 'data\\' + site_name + '\\' + tile_name + '\\'
    path_dst = 'data\\' + tile_name + '\\'

    ## if no testing data ###
    if test_prop == 0:
        # split
        tile_list_train, tile_list_val = train_test_split(tile_list, test_size=val_prop, random_state=seed)
        # write
        for file in tile_list_train:
            copy( path_src + 'img\\tile\\' + file, path_dst + path_train_img)
            copy( path_src + 'msk\\tile\\' + file, path_dst + path_train_msk)
        for file in tile_list_val:
            copy( path_src + 'img\\tile\\' + file, path_dst + path_val_img)
            copy( path_src + 'msk\\tile\\' + file, path_dst + path_val_msk)
        # print
        print(site_name + ' split: ' + 'TRAINING=' + str(1-val_prop) + ', VALIDATION=' + str(val_prop))

    ### if all testing data ###
    elif test_prop == 1:
        # write
        for file in tile_list:
            copy( path_src + 'img\\tile\\' + file, path_dst + path_test_img)
            copy( path_src + 'msk\\tile\\' + file, path_dst + path_test_msk)
        # print
        print(site_name + ' split: ' + 'TESTING=' + str(1))

    ### if both testing and training ###
    else:
        # split
        tile_list_trainval, tile_list_test = train_test_split(tile_list, test_size=test_prop, random_state=seed)
        tile_list_train, tile_list_val = train_test_split(tile_list_trainval, test_size=val_prop, random_state=seed)
        # write
        for file in tile_list_train:
            copy( path_src + 'img\\tile\\' + file, path_dst + path_train_img)
            copy( path_src + 'msk\\tile\\' + file, path_dst + path_train_msk)
        for file in tile_list_val:
            copy( path_src + 'img\\tile\\' + file, path_dst + path_val_img)
            copy( path_src + 'msk\\tile\\' + file, path_dst + path_val_msk)
        for file in tile_list_test:
            copy(path_src + 'img\\tile\\' + file, path_dst + path_test_img)
            copy(path_src + 'msk\\tile\\' + file, path_dst + path_test_msk)
        # print
        print(site_name + ' split: ' +
              'TRAINING=' + str(round(1 - (test_prop + (val_prop * (1 - test_prop))), 2)) +
              ', VALIDATION=' + str(round(val_prop * (1 - test_prop), 2)) +
              ', TESTING=' + str(test_prop))

########################################################################################################################
### move training, validation and testing data to common folder for fitting  ###
########################################################################################################################
def copy_data(site_name=None,tile_size=None,folder_name=None):
    # get paths
    tile_name = 'tile_' + str(tile_size)
    path = 'data\\' + site_name + '\\' + tile_name + '\\'
    path_src = path + folder_name + '\\'
    path_dst = 'data\\' + tile_name + '\\' + folder_name + '\\'
    # write
    [copy(path_src + i, path_dst + i) for i in os.listdir(path_src)]


########################################################################################################################
########################################################################################################################
################################################################################################################
### MODEL_FIT functions ###
################################################################################################################
########################################################################################################################
########################################################################################################################

########################################################################################################################
### downsample ###
########################################################################################################################
def pool2d(A, kernel_size, stride, padding=0, pool_mode='max'):
    '''
    written by Andreas K.(https://stackoverflow.com/users/2945357/andreas-k)
     2D Pooling

     Parameters:
         A: input 2D array
         kernel_size: int, the size of the window over which we take pool
         stride: int, the stride of the window
         padding: int, implicit zero paddings on both sides of the input
         pool_mode: string, 'max' or 'avg'
     '''
    # Padding
    A = np.pad(A, padding, mode='constant')

    # Window view of A
    output_shape = ((A.shape[0] - kernel_size) // stride + 1,
                    (A.shape[1] - kernel_size) // stride + 1)

    shape_w = (output_shape[0], output_shape[1], kernel_size, kernel_size)
    strides_w = (stride * A.strides[0], stride * A.strides[1], A.strides[0], A.strides[1])

    A_w = as_strided(A, shape_w, strides_w)

    # Return the result of pooling
    if pool_mode == 'max':
        return A_w.max(axis=(2, 3))
    elif pool_mode == 'avg':
        return A_w.mean(axis=(2, 3))

########################################################################################################################
### downsample ###
########################################################################################################################
def downsample(A=None, kernel_size=2, stride=2, padding=0, pool_mode='max', inter_method='AREA'):
    # downsample each channel
    Apool = [pool2d(A[:, :, i], kernel_size=kernel_size, stride=stride, padding=padding, pool_mode=pool_mode) for i in
             range(A.shape[2])]
    # stack channels
    Apool = np.stack(Apool, axis=2)
    # upsample with area interpolation to original dimension of image
    if inter_method == 'AREA':
        Apool = cv2.resize(Apool, dsize=A.shape[0:2], interpolation=cv2.INTER_AREA)
    elif inter_method == 'NEAREST':
        Apool = cv2.resize(Apool, dsize=A.shape[0:2], interpolation=cv2.INTER_NEAREST)
    # to integer
    Apool = Apool.astype('uint8')
    # if single channel image then reshape
    if len(Apool.shape) == 2:
        Apool = Apool.reshape(Apool.shape[0], Apool.shape[1], 1)
    # return
    return Apool


########################################################################################################################
### construct generator ###
########################################################################################################################
class data_generator(keras.utils.Sequence):

    ############################################################################################################
    ### initialize ###
    ############################################################################################################
    def __init__(self, dat, x_path, y_path, batch_size, n_class, backbone, nchannels, class_names,
                 BRIGHT, BRIGHT_mul, BRIGHT_add,
                 SATURATION, SATURATION_mul, SATURATION_add,
                 TEMPERATURE, TEMPERATURE_ratio, TEMPERATURE_range,
                 random_order,
                 ROTATE, ROTATE_range, ROTATE_90,
                 DOWNSAMPLE, DOWNSAMPLE_kernel, DOWNSAMPLE_prop,
                 CROP, CROP_dim,
                 SCALE_FUN):
        # data
        self.dat = dat # tile names
        self.x_path = x_path # path to imgs
        self.y_path = y_path # path to msks
        self.batch_size = batch_size
        self.n_class = n_class
        self.backbone = backbone
        self.nchannels = nchannels
        self.class_names = class_names # classes to include, discarded classes set to 0 (BACKGROUND)
        # SPECTRAL AUGMENTER
        self.BRIGHT = BRIGHT # T/F
        self.BRIGHT_mul = BRIGHT_mul
        self.BRIGHT_add = BRIGHT_add
        self.SATURATION = SATURATION # T/F
        self.SATURATION_mul = SATURATION_mul
        self.SATURATION_add = SATURATION_add
        self.TEMPERATURE = TEMPERATURE # T/F
        self.TEMPERATURE_range = TEMPERATURE_range
        self.TEMPERATURE_ratio = TEMPERATURE_ratio
        self.random_order = random_order # T/F
        # ROTATE_radian AUGMENTER
        self.ROTATE = ROTATE # T/F
        self.ROTATE_range = ROTATE_range
        # ROTATE_90 AUGMENTER
        self.ROTATE_90 = ROTATE_90
        # DOWNSAMPLE AUGMENTER
        self.DOWNSAMPLE = DOWNSAMPLE # T/F
        self.DOWNSAMPLE_kernel = DOWNSAMPLE_kernel # proportion of images to downsample
        self.DOWNSAMPLE_prop = DOWNSAMPLE_prop
        # CROP
        self.CROP = CROP
        self.CROP_dim = CROP_dim
        self.SCALE_FUN = SCALE_FUN

    ############################################################################################################
    ### batches per epoch ###
    ############################################################################################################
    def __len__(self):
        return len(self.dat) // self.batch_size

    ############################################################################################################
    ### shuffle indices after each epoch ###
    ############################################################################################################
    def on_epoch_end(self):
        self.dat = shuffle(self.dat)

    ############################################################################################################
    ### AUGMENTER ###
    ############################################################################################################
    ############################################################
    ### saturation, brightness and temperature augmentation ###
    ############################################################
    def __augmenter_spectral(self):
        # BRIGHTNESS
        if self.BRIGHT == True and self.SATURATION == False and self.TEMPERATURE == False:
            return iaa.MultiplyAndAddToBrightness(mul=self.BRIGHT_mul, add=self.BRIGHT_add)

        # SATURATION
        elif self.BRIGHT == False and self.SATURATION == True and self.TEMPERATURE == False:
            return iaa.Sequential([iaa.MultiplySaturation(self.SATURATION_mul), iaa.AddToSaturation(self.SATURATION_add)],
                                 random_order=self.random_order)

        # TEMPERATURE
        elif self.BRIGHT == False and self.SATURATION == False and self.TEMPERATURE == True:
            if TEMPERATURE_ratio < 1:
                return iaa.Sometimes(self.TEMPERATURE_ratio, iaa.ChangeColorTemperature(self.TEMPERATURE_range))
            else:
                return iaa.ChangeColorTemperature(self.TEMPERATURE_range)

        # BRIGHTNESS & SATURATION
        elif self.BRIGHT == True and self.SATURATION == True and self.TEMPERATURE == False:
            return iaa.Sequential([iaa.MultiplyAndAddToBrightness(mul=self.BRIGHT_mul, add=self.BRIGHT_add),
                                  iaa.MultiplySaturation(self.SATURATION_mul), iaa.AddToSaturation(self.SATURATION_add)],
                                 random_order=self.random_order)

        # BRIGHTNESS & TEMPERATURE
        elif self.BRIGHT == True and self.SATURATION == False and self.TEMPERATURE == True:
            if self.TEMPERATURE_ratio < 1:
                return iaa.Sequential([iaa.MultiplyAndAddToBrightness(mul=self.BRIGHT_mul, add=self.BRIGHT_add),
                                      iaa.Sometimes(self.TEMPERATURE_ratio, iaa.ChangeColorTemperature(self.TEMPERATURE_range))],
                                     random_order=self.random_order)
            else:
                return iaa.Sequential([iaa.MultiplyAndAddToBrightness(mul=self.BRIGHT_mul, add=self.BRIGHT_add),
                                      iaa.ChangeColorTemperature(self.TEMPERATURE_range)],
                                     random_order=self.random_order)

        # SATURATION & TEMPERATURE
        elif self.BRIGHT == False and self.SATURATION == True and self.TEMPERATURE == True:
            if self.TEMPERATURE_ratio < 1:
                return iaa.Sequential([iaa.MultiplySaturation(self.SATURATION_mul), iaa.AddToSaturation(self.SATURATION_add),
                                      iaa.Sometimes(self.TEMPERATURE_ratio, iaa.ChangeColorTemperature(self.TEMPERATURE_range))],
                                     random_order=self.random_order)
            else:
                return iaa.Sequential([iaa.MultiplySaturation(self.SATURATION_mul), iaa.AddToSaturation(self.SATURATION_add),
                                      iaa.ChangeColorTemperature(self.TEMPERATURE_range)],
                                     random_order=self.random_order)

        # BRIGHTNESS & SATURATION & TEMPERATURE
        elif self.BRIGHT == True and self.SATURATION == True and self.TEMPERATURE == True:
            if self.TEMPERATURE_ratio < 1:
                return iaa.Sequential([iaa.MultiplyAndAddToBrightness(mul=self.BRIGHT_mul, add=self.BRIGHT_add),
                                      iaa.MultiplySaturation(self.SATURATION_mul), iaa.AddToSaturation(self.SATURATION_add),
                                      iaa.Sometimes(self.TEMPERATURE_ratio, iaa.ChangeColorTemperature(self.TEMPERATURE_range))],
                                     random_order=self.random_order)
            else:
                return iaa.Sequential([iaa.MultiplyAndAddToBrightness(mul=self.BRIGHT_mul, add=self.BRIGHT_add),
                                      iaa.MultiplySaturation(self.SATURATION_mul), iaa.AddToSaturation(self.SATURATION_add),
                                      iaa.ChangeColorTemperature(self.TEMPERATURE_range)],
                                     random_order=self.random_order)

    ############################################################
    ### EIGHT TRANSFORMATIONS BASED ON FLIPS AND ROTATE_90  ###
    ############################################################
    def rotate_90(self, x=None, type=None):
        if type == 0:
            return x
        elif type == 1:
            return np.asarray(tf.image.flip_left_right(x))
        elif type == 2:
            return np.asarray(tf.image.flip_up_down(x))
        elif type == 3:
            return np.asarray(tf.image.rot90(tf.image.flip_left_right(x), 1))
        elif type == 4:
            return np.asarray(tf.image.rot90(tf.image.flip_up_down(x), 1))
        elif type == 5:
            return np.asarray(tf.image.rot90(x, 1))
        elif type == 6:
            return np.asarray(tf.image.rot90(x, 2))
        elif type == 7:
            return np.asarray(tf.image.rot90(x, 3))

    ############################################################
    ### AUGMENTER ###
    ############################################################

    def __augmenter(self, img, msk):
        # saturation, brightness and temperature augmentation
        if self.BRIGHT or self.SATURATION or self.TEMPERATURE:
            seq = self.__augmenter_spectral()
            img = np.array([seq(image=img[i]) for i in range(img.shape[0])])

        # downsample
        if self.DOWNSAMPLE:
            # generate random indices for downsampling
            down = random.sample(range(self.batch_size), int(np.floor(self.batch_size * self.DOWNSAMPLE_prop)))
            # randomly sample DOWNSAMPLE_prop of images for downsampling
            for i in down:
                img[i] = downsample(img[i], kernel_size=self.DOWNSAMPLE_kernel, stride=self.DOWNSAMPLE_kernel,
                                    padding=0, pool_mode='avg', inter_method='AREA')
                msk[i] = downsample(msk[i], kernel_size=self.DOWNSAMPLE_kernel, stride=self.DOWNSAMPLE_kernel, padding=0,
                                    pool_mode='max', inter_method='NEAREST')


        # rotate images and masks
        if self.ROTATE:
            rot = random.choices(range(self.ROTATE_range[0], self.ROTATE_range[1]), k=self.batch_size)
            for i in range(img.shape[0]):
                img[i] = np.asarray(tfa.image.rotate(img[i], angles=rot[i], interpolation='bilinear'))
                msk[i] = np.asarray(tfa.image.rotate(msk[i], angles=rot[i], interpolation='nearest')).reshape(
                    msk[i].shape[0], msk[i].shape[1], 1)

        # rotate 90
        if self.ROTATE_90:
            rot_90 = random.choices(range(7), k=self.batch_size)
            for i in range(img.shape[0]):
                img[i] = self.rotate_90(img[i], rot_90[i])
                msk[i] = self.rotate_90(msk[i], rot_90[i])

        # return
        return img, msk

    ############################################################################################################
    ### DATA GENERATOR ###
    ############################################################################################################

    # data generation - load images and masks associated with IDs, image augmentation and preprocessing
    def __datagen(self, batch):
        ### LOAD ###
        img = np.array([load_tile(self.x_path + file, self.nchannels) for file in batch])
        msk = np.array([load_tile(self.y_path + file, 1) for file in batch])
        ### REMOVE UNUSED CLASSES (SET TO BACKGROUND) ###
        n, row, col = msk.shape
        msk = msk.reshape(n * row * col)
        msk = np.where(np.in1d(msk, self.class_names), msk, 0)
        msk = msk.reshape(n, row, col)
        del n, row, col
        ### RESHAPE ###
        img = img.reshape(img.shape[0], img.shape[1], img.shape[2], self.nchannels)
        msk = msk.reshape(msk.shape[0],msk.shape[1],msk.shape[2],1)
        ### AUGMENT ###
        img, msk = self.__augmenter(img,msk)
        ### PREPROCESS ###
        preprocess_input = sm.get_preprocessing(self.backbone)
        if self.SCALE_FUN == '1/255':
            img = img/255
        elif self.SCALE_FUN == 'MinMaxScaler':
            img = [MinMaxScaler().fit_transform(tile.reshape(-1, tile.shape[-1])).reshape(tile.shape) for tile in img]
            img = np.asarray(img)
        img = img.astype('float32')
        img = preprocess_input(img)
        msk = ku.to_categorical(msk, self.n_class)
        if self.CROP:
            # crop msk and convert to tensor
            msk = kl.Cropping2D(cropping=((self.CROP_dim,self.CROP_dim),(self.CROP_dim,self.CROP_dim)))(msk)
        else:
            msk = tf.convert_to_tensor(msk)
        img = tf.convert_to_tensor(img)
        # return
        return img, msk

    ############################################################################################################
    ### return complete batch ###
    ############################################################################################################
    def __getitem__(self, id):
        # get batch IDs
        batch = self.dat[id * self.batch_size:(id + 1) * self.batch_size]
        # process data
        out_x, out_y = self.__datagen(batch)
        # return
        return out_x, out_y

########################################################################################################################
########################################################################################################################
################################################################################################################
### MODEL_PREDICT functions ###
################################################################################################################
########################################################################################################################
########################################################################################################################

########################################################################################################################
### preprocess data to fit model architecture ###
########################################################################################################################
def preprocess_img(img=None, backbone=None,SCALE_FUN=None):
    preprocess = sm.get_preprocessing(backbone)
    if SCALE_FUN == '1/255':
        img = img / 255
    elif SCALE_FUN == 'MinMaxScaler':
        img = MinMaxScaler().fit_transform(img.reshape(-1, img.shape[-1])).reshape(img.shape)
    img = img.astype('float32')
    img = preprocess(img)
    return img

########################################################################################################################
### predict and mosaic ###
########################################################################################################################
def predict_mosaic(site_name=None, model_name=None, backbone=None, tile_size=None, pad=None, nchannels=None, BATCH=None,
                   CROP=None,CROP_dim=None,SCALE_FUN=None):
    ### SET PATHS ###
    tile_name = 'tile_' + str(tile_size)
    img_path = 'data\\' + site_name + '\\' + tile_name + '\\img\\full\\img.tif'
    model_path = 'data\\' + tile_name + '\\model\\' + model_name + '.hdf5'
    path_out = 'data\\' + tile_name + '\\run\\' + model_name + '\\' + 'output\\' + site_name + '_predict_msk.tif'

    ### LOAD ###
    img = gdal.Open(img_path)
    PROJ = img.GetProjection()  # get projection
    AFFINE = img.GetGeoTransform()  # get affine
    model = load_model(model_path, compile=False)
    # remove projection
    img = [img.GetRasterBand(i).ReadAsArray() for i in range(1, nchannels + 1)]
    img = np.stack(img, axis=2)
    # to 8bit
    img = img.astype(np.uint8)
    ### patchify image ###
    img_patch = patchify.patchify(img, (tile_size, tile_size, nchannels), pad)
    nrow = img_patch.shape[0]
    ncol = img_patch.shape[1]
    n_samples = int(nrow * ncol)
    # flatten
    img_patch = img_patch.reshape(nrow * ncol, tile_size, tile_size, nchannels)
    # preprocess
    img_patch = [preprocess_img(img, backbone=backbone,SCALE_FUN=SCALE_FUN) for img in img_patch]
    img_patch = np.array(img_patch)
    #img_patch = tf.convert_to_tensor(img_patch)

    ### PREDICT ###
    out = []
    # batch
    for i in range(BATCH, ((n_samples // BATCH) * BATCH) + BATCH, BATCH):
        out_temp = model.predict(img_patch[(i - BATCH):i])
        out_temp = np.argmax(out_temp, axis=3)
        out_temp = out_temp.astype('uint8')
        out.append(out_temp)
        del out_temp
        gc.collect()
    # final batch
    if (n_samples - (n_samples // BATCH) * BATCH) != 0:
        out_temp = model.predict(img_patch[(n_samples // BATCH) * BATCH:n_samples])
        out_temp = np.argmax(out_temp, axis=3)
        out_temp = out_temp.astype('uint8')
        out.append(out_temp)
        del out_temp
        gc.collect()
    # flatten to numpy array
    out_array = np.concatenate(out)

    ### MOSAIC ###
    # if cropping is built into CNN
    if CROP:
        tile_out = out_array.reshape(nrow, ncol, pad, pad)
        tile_out = tile_out.transpose(0, 2, 1, 3)
        tile_out = tile_out.reshape(nrow * pad, ncol * pad)
        # pad
        tile_out = np.pad(tile_out,
                          ((CROP_dim, (img.shape[0] - CROP_dim) - tile_out.shape[0]),
                           (CROP_dim, (img.shape[1] - CROP_dim) - tile_out.shape[1]))
                          )
    # otherwise crop now
    else:
        out_array_clip = np.array(
            [A[int(pad / 2):int(pad + pad / 2), int(pad / 2):int(pad + pad / 2)] for A in out_array])
        # reshape
        tile_out = out_array_clip.reshape(nrow, ncol, pad, pad)
        tile_out = tile_out.transpose(0, 2, 1, 3)
        tile_out = tile_out.reshape(nrow * pad, ncol * pad)
        # pad
        tile_out = np.pad(tile_out, int(pad / 2))

    ### WRITE ###
    # write data (non-geospatial)
    cv2.imwrite(path_out, tile_out)
    # assign projection
    predict_msk = gdal.Open(path_out)
    predict_msk = predict_msk.SetProjection(PROJ)
    predict_msk = None
    # assign affine
    predict_msk = gdal.Open(path_out)
    predict_msk.SetGeoTransform(AFFINE)
    predict_msk = None

    print(site_name + ' PREDICTED and mosaiced')



########################################################################################################################
########################################################################################################################
################################################################################################################
### MODEL_TEST functions ###
################################################################################################################
########################################################################################################################
########################################################################################################################

### calculate metrics ###
def metric_calc(y_true = None, y_pred = None, num_class = None, avg = None, lab=None):
    # create empty array to store metric calculations
    if avg == None:
        metric_list = np.empty(shape=(4, y_true.shape[0], num_class), dtype="float64")
    else:
        metric_list = np.empty(shape=(4, y_true.shape[0], 1), dtype="float64")

    # iterate across tiles and calculate metrics
    for i in range(y_true.shape[0]):
        # flatten data
        yt = y_true[i,:,:].flatten()
        yp = y_pred[i,:,:].flatten()
        # calculate scores
        metric_list[0, i, :] = metrics.jaccard_score(yt, yp, average=avg, labels=lab, zero_division=0)
        metric_list[1, i, :] = metrics.f1_score(yt, yp, average=avg, labels=lab, zero_division=0)
        metric_list[2, i, :] = metrics.precision_score(yt, yp, average=avg, labels=lab, zero_division=0)
        metric_list[3, i, :] = metrics.recall_score(yt, yp, average=avg, labels=lab, zero_division=0)

    # set all negatives to nan
    metric_list[metric_list == 0] = np.nan

    # return
    return metric_list
### calculate metrics ###
def metric_calc2(y_true = None, y_pred = None, num_class = None, avg = None, lab=None):
    # create empty array to store metric calculations
    if avg == None:
        metric_list = np.zeros(shape=(4, 1, num_class), dtype="float64")
    else:
        metric_list = np.zeros(shape=(4, 1, 1), dtype="float64")

    # calculate scores
    metric_list[0, 0, :] = metrics.jaccard_score(y_true, y_pred, average=avg, labels=lab, zero_division=1)
    metric_list[1, 0, :] = metrics.f1_score(y_true, y_pred, average=avg, labels=lab, zero_division=1)
    metric_list[2, 0, :] = metrics.precision_score(y_true, y_pred, average=avg, labels=lab, zero_division=1)
    metric_list[3, 0, :] = metrics.recall_score(y_true, y_pred, average=avg, labels=lab, zero_division=1)

    # set all negatives to nan
    metric_list[metric_list == 1] = np.nan

    # return
    return metric_list