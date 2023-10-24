########################################################################################################################
start = time.process_time()
########################################################################################################################

########################################################################################################################
########################################################################################################################
################################################################################################################
### set arguments ###
########################################################################################################################
########################################################################################################################
################################################################################################################
# user defined
model_name = 'PLACEHOLDER'
backbone = 'resnet50'
SCALE_FUN = 'MinMaxScaler'
nchannels = 3
tile_size = 128
CROP=False
CROP_dim = 16
# batch size used for prediction
BATCH=500
if CROP:
    pad = tile_size - (CROP_dim * 2)
else:
    pad = 64
# seed - use in testing
seed = 0

# site name
site_name = ['site2_h15m_0913', 'site5_h7m_0906', 'site6_h15m_0907', 'site6_h30m_0907', 'site8_h7m_0905',
              'site9_h7m_0905', 'site10_h7m_0905', 'site15_h15m_0907', 'siteA_h15m_0913', 'siteA_h30m_0913',
              'siteCF_h15m_0907', 'siteCF_h30m_0907']

# auto
tile_name = 'tile_' + str(tile_size)
model_path = 'data\\' + tile_name + '\\model\\' + model_name + '.hdf5'
full_path = ['data/'+tile_name+'/run/'+model_name+'/output/'+i+'_predict_msk.tif' for i in site_name]
tile_path = 'data/'+tile_name+'/run/'+model_name+'/output/tile/'

# create directories
mk_dir('data\\' + tile_name + '\\run\\' + model_name)
mk_dir('data\\' + tile_name + '\\run\\' + model_name + '\\' + 'output\\')
mk_dir('data\\' + tile_name + '\\run\\' + model_name + '\\' + 'output\\tile\\')

################################################################################################################
### run ###
################################################################################################################

########################
### predict ###
########################
[predict_mosaic(i, model_name, backbone, tile_size, pad, nchannels, BATCH, CROP, CROP_dim, SCALE_FUN) for i in site_name]

print(time.process_time() - start)
########################################################################################################################

########################
### grid predictions ###
########################
[tile_raster2(full_path=full_path[i], tile_path=tile_path[i], site_name=site_name[i], tile_size=tile_size, nchannels=1) for i in range(len(site_name))]
[tile_raster2(full_path=full_path[i], tile_path=tile_path, site_name=site_name[i], tile_size=tile_size, nchannels=1) for i in range(len(site_name))]

print(time.process_time() - start)
########################################################################################################################