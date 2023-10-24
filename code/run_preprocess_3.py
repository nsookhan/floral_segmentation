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
# Training site(s): train, validation, test split proportions
test_prop_TRAIN = 0.2 #0
val_prop_TRAIN = 0.25 #0.2
# Testing site(s): train, validation, test split proportions
test_prop_TEST = 1
val_prop_TEST = 0
# Fine-tuning site(s): train, validation, test split proportions. These are sites that require fine-tuning
test_prop_FINE = 0.2
val_prop_FINE = 0.25
# seed - use in testing
seed = 0

### set training and testing ###
# all site names
site_name = ['site2_h15m_0913', 'site5_h7m_0906', 'site6_h15m_0907', 'site6_h30m_0907', 'site8_h7m_0905',
             'site9_h7m_0905', 'site10_h7m_0905', 'site15_h15m_0907', 'siteA_h15m_0913', 'siteA_h30m_0913',
             'siteCF_h15m_0907', 'siteCF_h30m_0907']
# sites used for training
train_site = ['site2_h15m_0913', 'site5_h7m_0906', 'site6_h15m_0907', 'site8_h7m_0905',
              'site9_h7m_0905', 'site10_h7m_0905', 'site15_h15m_0907', 'siteA_h15m_0913',
              'siteCF_h15m_0907', 'site6_h30m_0907', 'siteA_h30m_0913', 'siteCF_h30m_0907']
# sites used for testing
test_site = ['PLACE_HOLDER']

# sites used for fine-tuning
fine_site = ['PLACE_HOLDER']

# create directory for training and testing
trainvaltest_dir(tile_size)

########################################################################################################################
########################################################################################################################
################################################################################################################
### RUN ###
########################################################################################################################
########################################################################################################################
################################################################################################################
### train, validation, test split ###
# load tile list of masked tiles for each site
tile_list = [load_tile_list(site_name=i,tile_size=tile_size,folder_name='grd',project_name=project_name) for i in site_name]

# split
for i in range(len(site_name)):
    # TRAINING sites
    if site_name[i] in train_site:
        train_val_test_split(site_name=site_name[i], tile_size=tile_size, tile_list = tile_list[i],
                             test_prop=test_prop_TRAIN, val_prop=val_prop_TRAIN, seed=seed,
                             path_train_img='train_img\\train\\', path_train_msk='train_msk\\train\\',
                             path_val_img='val_img\\val\\', path_val_msk='val_msk\\val\\',
                             path_test_img='test_img\\test\\', path_test_msk='test_msk\\test\\')
    # TEST sites
    elif site_name[i] in test_site:
        train_val_test_split(site_name=site_name[i], tile_size=tile_size, tile_list = tile_list[i],
                             test_prop=test_prop_TEST, val_prop=val_prop_TEST, seed=seed,
                             path_train_img='train_img\\train\\', path_train_msk='train_msk\\train\\',
                             path_val_img='val_img\\val\\', path_val_msk='val_msk\\val\\',
                             path_test_img='test_img\\test\\', path_test_msk='test_msk\\test\\')
    # FINE-TUNE sites
    elif site_name[i] in fine_site:
        train_val_test_split(site_name=site_name[i], tile_size=tile_size, tile_list = tile_list[i],
                             test_prop=test_prop_FINE, val_prop=val_prop_FINE, seed=seed,
                             path_train_img='train2_img\\train\\', path_train_msk='train2_msk\\train\\',
                             path_val_img='val2_img\\val\\', path_val_msk='val2_msk\\val\\',
                             path_test_img='test_img\\test\\', path_test_msk='test_msk\\test\\')
    # site not included in analysis
    else:
        print(site_name + ' NOT INCLUDED IN ANALYSIS')