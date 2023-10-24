####################################################
################################################################################################################
### set paths ###
################################################################################################################
tile_size = 128
BACKBONE = 'resnet50'
model_name = '20230228_RESNET50_1'
split_number = '__SPLIT5'
project_name = 'floral_segmentation'
n_class = 4
# class_names = [0, 1, 2, 3, 4, 5]
# seed - use in testing
seed = 0

### set site names ###
site_name = [
    'site2_h15m_0913', #'site2_h15m_1012',
    'site5_h7m_0906', #'site5_h7m_1011',
    'site6_h15m_0907', 'site6_h30m_0907', #'site6_h15m_1011',
    'site8_h7m_0905',
    'site9_h7m_0905', #'site9_h7m_1012',
    'site10_h7m_0905',
    'site15_h15m_0907',
    'siteA_h15m_0913', 'siteA_h30m_0913', #'siteA_h15m_1012',
    'siteCF_h15m_0907', 'siteCF_h30m_0907', #'siteCF_h15m_1011'
    ]
# sites used for testing
test_site = [
    'site2_h15m_0913', #'site2_h15m_1012',
    'site5_h7m_0906', #'site5_h7m_1011',
    'site6_h15m_0907', 'site6_h30m_0907', #'site6_h15m_1011',
    'site8_h7m_0905',
    'site9_h7m_0905', #'site9_h7m_1012',
    'site10_h7m_0905',
    'site15_h15m_0907',
    'siteA_h15m_0913', 'siteA_h30m_0913', #'siteA_h15m_1012',
    'siteCF_h15m_0907', 'siteCF_h30m_0907', #'siteCF_h15m_1011'
    ]
# test_site = ['h7m','h15m','h30m']

# auto
tile_name = 'tile_' + str(tile_size)
path_true = 'data\\' + tile_name + '\\' + split_number + '\\test_msk\\test\\'
path_pred = 'data\\' + tile_name + '\\run\\' + model_name + '\\output\\tile\\'
metric_out_path = 'data\\' + tile_name + '\\run\\' + model_name + '\\metric\\'
mk_dir(metric_out_path)

################################################################################################################
### run ###
################################################################################################################

########################
### get tiles ###
########################
# sort tiles by site
tile_site = [[site in tile for tile in os.listdir(path_true)] for site in test_site]
tile_site = [list(compress(os.listdir(path_true),site)) for site in tile_site]
### load tiles (ground truth)
y_true = [np.array([load_tile(path_true+tile,1) for tile in site]).flatten() for site in tile_site]
### load tiles (prediction)
y_pred = [np.array([load_tile(path_pred+tile,1) for tile in site]).flatten() for site in tile_site]

########################
### calculate metrics ###
########################
### by sites and calculate metrics
metric_val = [metric_calc2(y_true=y_true[i], y_pred=y_pred[i], num_class=n_class, avg=None, lab = [0, 1, 2, 3]) for i in range(len(test_site))]
metric_val_w = [metric_calc2(y_true=y_true[i], y_pred=y_pred[i], num_class=n_class, avg='weighted', lab = [1, 2, 3]) for i in range(len(test_site))]

########################
### calculate confusion matrix ###
########################
### by site
c = [metrics.multilabel_confusion_matrix(y_true[i],y_pred[i]) for i in range(len(test_site))]

########################
### write to disk ###
########################
### by site
for i in range(len(test_site)):
    # by class, by grid
    [np.save(file=metric_out_path + test_site[i] + '_metric_class_grid_' + str(j) + '.npy', arr=metric_val[i][j]) for j in range(metric_val[i].shape[0])]
    # weighted, by grid
    [np.save(file=metric_out_path + test_site[i] + '_metric_weighted_grid_' + str(j) + '.npy', arr=metric_val_w[i][j]) for j in range(metric_val_w[i].shape[0])]
    # confusion matrix
    np.save(file=metric_out_path + test_site[i] + '_confusion_matrix.npy', arr=c[i])

########################
### ALL ###
########################
### get tiles ###
y_true_all = reduce(lambda a, b: np.append(a,b, axis=0), list(compress(y_true, [i in test_site for i in site_name])))
y_pred_all = reduce(lambda a, b: np.append(a,b, axis=0), list(compress(y_pred, [i in test_site for i in site_name])))
### calculate metrics ###
metric_val_all = metric_calc2(y_true=y_true_all, y_pred=y_pred_all, num_class=n_class, avg=None, lab=[0, 1, 2, 3])
metric_val_w_all = metric_calc2(y_true=y_true_all, y_pred=y_pred_all, num_class=n_class, avg='weighted', lab=[1, 2, 3])
### calculate confusion matrix ###
# c_all = metrics.multilabel_confusion_matrix(y_true_all,y_pred_all)
c_all = metrics.confusion_matrix(y_true_all,y_pred_all)
### write to disk ###
# by class
[np.save(file=metric_out_path + 'metric_class_grid_' + str(i) + '.npy', arr=metric_val_all[i]) for i in range(4)]
# across class
[np.save(file=metric_out_path + 'metric_weighted_grid_' + str(i) + '.npy', arr=metric_val_w_all[i]) for i in range(4)]
# confusion matrix
np.save(file=metric_out_path + "confusion_matrix.npy", arr=c_all)