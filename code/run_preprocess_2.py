########################################################################################################################
########################################################################################################################
################################################################################################################
### set arguments ###
########################################################################################################################
########################################################################################################################
################################################################################################################

### model parameters ###
project_name = 'floral_segmentation'
tile_size = 128
nchannels = 3

### set site names ###
# all site names
site_name = ['site2_h15m_0913', 'site5_h7m_0906', 'site6_h15m_0907', 'site6_h30m_0907', 'site8_h7m_0905',
             'site9_h7m_0905', 'site10_h7m_0905', 'site15_h15m_0907', 'siteA_h15m_0913', 'siteA_h30m_0913',
             'siteCF_h15m_0907', 'siteCF_h30m_0907']
# path to manual masks
msk_path = ['20210913_site2_h15m_msk.shp', '20210906_site5_h7m_msk.shp', '20210907_site6_h15m_msk.shp', '20210907_site6_h30m_msk.shp', '20210905_site8_h7m_msk.shp',
            '20210905_site9_h7m_msk.shp', '20210905_site10_h7m_msk.shp', '20210907_site15_h15m_msk.shp', '20210913_siteA_h15m_msk.shp', '20210913_siteA_h30m_msk.shp',
            '20210907_siteCF_h15m_msk.shp', '20210907_siteCF_h30m_msk.shp']
# path to manual mask grids
grd_path = ['20210913_site2_h15m_grd.shp', '20210906_site5_h7m_grd.shp', '20210907_site6_h15m_grd.shp', '20210907_site6_h30m_grd.shp','20210905_site8_h7m_grd.shp',
            '20210905_site9_h7m_grd.shp', '20210905_site10_h7m_grd.shp', '20210907_site15_h15m_grd.shp','20210913_siteA_h15m_grd.shp','20210913_siteA_h30m_grd.shp',
            '20210907_siteCF_h15m_grd.shp', '20210907_siteCF_h30m_grd.shp']

### auto ###
# path to orthomosaics
img_path = ['data/' + site + '/' + 'tile_' + str(tile_size) + '/img/full/img.tif' for site in site_name]
msk_path_dir = '__raw/msk/'
grd_path_dir = '__raw/grd/'
msk_path = [msk_path_dir + img for img in msk_path]
grd_path = [grd_path_dir + img for img in grd_path]

########################################################################################################################
################################################################################################################
### RUN ###
########################################################################################################################
########################################################################################################################
################################################################################################################

### process mask ###
# MANUAL MASK #
# rasterize
[rasterize(site_name=site_name[i], tile_size=tile_size, folder_name='msk', msk_path=msk_path[i], img_path=img_path[i], attr='flow_cat') for i in range(len(site_name))]
# tile
[tile_raster(site_name=i,tile_size=tile_size,folder_name='msk',nchannels=1) for i in site_name]

# MANUAL MASK - GRID #
# rasterize
[rasterize(site_name=site_name[i], tile_size=tile_size, folder_name='grd', msk_path=grd_path[i], img_path=img_path[i], attr='msk') for i in range(len(site_name))]
# tile
[tile_raster(site_name=i,tile_size=tile_size,folder_name='grd',nchannels=1) for i in site_name]
# create list of manually masked tiles
[create_tile_list(site_name=i, tile_size=tile_size, folder_name='grd', project_name=project_name) for i in site_name]