########################################################################################################################
start = time.process_time()
########################################################################################################################
model_name = 'PLACEHOLDER'
split_number = '__SPLIT5'
tile_size = 128
encoder_freeze = 0.95
train_freeze = False # T/F train with encoder layers frozen before training with percent encoder layers released
loss_back = True #T/F include background in loss function
use_reg = False # regularization
reg_class = keras.regularizers.l2
reg_lambda = 0.001
# generator parameters
parameters_train = {
    'batch_size': 32,
    'n_class': 4,
    'backbone': 'resnet50',
    'nchannels': 3,
    'class_names': [1, 2, 3],
    'BRIGHT': True, 'BRIGHT_mul': (0.8, 1.1), 'BRIGHT_add': (-20, 15),
    'SATURATION': True, 'SATURATION_mul': (0.6, 1.4), 'SATURATION_add': (-5, 5),
    'TEMPERATURE': True, 'TEMPERATURE_ratio': 1, 'TEMPERATURE_range': (5000, 10000),
    'random_order': True,
    'ROTATE': False, 'ROTATE_range': (-360, 360),
    'ROTATE_90': True,
    'DOWNSAMPLE': False, 'DOWNSAMPLE_kernel': 2, 'DOWNSAMPLE_prop': 0.25,
    'CROP': False, 'CROP_dim': 16,
    'SCALE_FUN': 'MinMaxScaler'}
parameters_val = {
    'batch_size': 32,
    'n_class': 4,
    'backbone': 'resnet50',
    'nchannels': 3,
    'class_names': [1, 2, 3],
    'BRIGHT': True, 'BRIGHT_mul': (0.8, 1.1), 'BRIGHT_add': (-20, 15),
    'SATURATION': True, 'SATURATION_mul': (0.6, 1.4), 'SATURATION_add': (-5, 5),
    'TEMPERATURE': True, 'TEMPERATURE_ratio': 1, 'TEMPERATURE_range': (5000, 10000),
    'random_order': True,
    'ROTATE': False, 'ROTATE_range': (-360, 360),
    'ROTATE_90': True,
    'DOWNSAMPLE': False, 'DOWNSAMPLE_kernel': 2, 'DOWNSAMPLE_prop': 0.25,
    'CROP': False, 'CROP_dim': 16,
    'SCALE_FUN': 'MinMaxScaler'}


# auto
tile_name = 'tile_' + str(tile_size)
path = 'data/' + tile_name + '/'
# inputs (MAIN FIT)
train_img_path, train_msk_path = path + split_number + '/train_img/train/', path + split_number + '/train_msk/train/'
val_img_path, val_msk_path = path + split_number + '/val_img/val/', path + split_number + '/val_msk/val/'
train_names, val_names = [i for i in os.listdir(train_img_path)], [i for i in os.listdir(val_img_path)]
# inputs (FINE-TUNE FIT)
train2_img_path, train2_msk_path = path + split_number + '/train2_img/train/', path + split_number + '/train2_msk/train/'
val2_img_path, val2_msk_path = path + split_number + '/val2_img/val/', path + split_number + '/val2_msk/val/'
train2_names, val2_names = [i for i in os.listdir(train2_img_path)], [i for i in os.listdir(val2_img_path)]
# model path
model_path = path + '/model/'

################################################################################
### CONSTRUCT GENERATOR ###
################################################################################
# MAIN FIT
train_img_gen = data_generator(dat = train_names, x_path = train_img_path, y_path = train_msk_path, **parameters_train)
val_img_gen = data_generator(dat = val_names, x_path = val_img_path, y_path = val_msk_path, **parameters_val)

################################################################################
### CALLBACK ###
################################################################################
callback = EarlyStopping(monitor='val_loss', min_delta=0.0001, patience=20, verbose=0, restore_best_weights=True)

################################################################################
### LOSS ###
################################################################################
if loss_back:
    loss = sm.losses.categorical_focal_jaccard_loss
else:
    loss = sm.losses.CategoricalFocalLoss(class_indexes=parameters_train['class_names']) + \
           sm.losses.JaccardLoss(class_indexes=parameters_train['class_names'])

################################################################################
### COMPILE AND FIT MODEL ###
################################################################################
###################################
# freeze encoder
###################################
model = sm.Unet(parameters_train['backbone'], encoder_weights='imagenet',
                input_shape=train_img_gen[0][0].shape[1:],
                classes=parameters_train['n_class'], activation='softmax', encoder_freeze=True)

# regularization
if use_reg:
    #model = set_regularization(model, kernel_regularizer=reg_class(reg_lambda), bias_regularizer= reg_class(reg_lambda) )
    model = set_regularization(model, kernel_regularizer=reg_class(reg_lambda))

# crop output
if parameters_train['CROP']:
    decoder = model.layers[len(model.layers) - 1].output
    decoder = kl.Cropping2D(cropping=((parameters_train['CROP_dim'], parameters_train['CROP_dim']),
                                      (parameters_train['CROP_dim'], parameters_train['CROP_dim'])))(decoder)
    model = keras.models.Model(inputs=model.inputs, outputs=[decoder])

# train with all encoder layers frozen
if train_freeze:
    # compile model
    model.compile('Adam', loss=loss, metrics=[sm.metrics.iou_score])
    # train
    history_freeze = model.fit(train_img_gen, epochs=300, verbose=1, callbacks=[callback], validation_data=val_img_gen)

###################################
# unfreeze encoder
###################################
if encoder_freeze < 1:
    # get frozen encoder layers
    train = (np.array([(layer.trainable) for layer in model.layers]) == False).nonzero()
    nm = np.array([(layer.name) for layer in model.layers])[train]
    # get a percentage of layers
    nm = nm[int(np.floor(len(nm) * encoder_freeze)):]
    # unfreeze selected layers
    for layers in model.layers:
        if layers.name in nm: layers.trainable = True
    # compile model
    model.compile('Adam', loss=loss, metrics=[sm.metrics.iou_score])
    # fine-tune
    history_unfreeze = model.fit(train_img_gen, epochs=300, verbose=1, callbacks=[callback],
                                 validation_data=val_img_gen)

###################################
### SAVE MODEL ###
###################################
model.save(model_path+model_name + '.hdf5')

###################################
### SAVE PLOT ###
###################################
try:
    history_freeze
except:
    iou, val_iou = history_unfreeze.history['iou_score'], history_unfreeze.history['val_iou_score']
    loss, val_loss = history_unfreeze.history['loss'], history_unfreeze.history['val_loss']
    history_output = np.transpose(np.vstack((iou, val_iou, loss, val_loss)))
    freeze_len = 0
else:
    iou, val_iou = history_freeze.history['iou_score'] + history_unfreeze.history['iou_score'],\
                   history_freeze.history['val_iou_score'] + history_unfreeze.history['val_iou_score']
    loss, val_loss = history_freeze.history['loss'] + history_unfreeze.history['loss'],\
                     history_freeze.history['val_loss'] + history_unfreeze.history['val_loss']
    freeze_len = len(history_freeze.history['iou_score'])
    fit_number = np.zeros(len(iou))
    fit_number[freeze_len:] = 1
    history_output = np.transpose(np.vstack((fit_number, iou, val_iou, loss, val_loss)))


fig, ((ax1), (ax2)) = plt.subplots(2,1, constrained_layout=True)
#plt.title('IoU Score')
ax1.plot(iou)
ax1.plot(val_iou)
ax1.set_ylabel('IoU')
ax1.set_xlabel('Epoch')
ax1.legend(['Training', 'Validation'], loc='lower right')
if freeze_len > 0: ax1.axline((freeze_len, min(iou)), (freeze_len, max(iou)),linestyle='--', color='black')
#plt.title('Categorical Focal Jaccard Loss')
ax2.plot(loss)
ax2.plot(val_loss)
ax2.set_ylabel('Loss')
ax2.set_xlabel('Epoch')
ax2.legend(['Training', 'Validation'], loc='upper right')
if freeze_len > 0: plt.axline((freeze_len, min(loss)), (freeze_len, max(loss)),linestyle='--', color='black')
# write
plt.savefig(model_path+model_name+'_plot.png', pad_inches=1)
np.save(file=model_path+model_name + '_history.npy', arr=history_output)


########################################################################################################################
print(time.process_time() - start)
########################################################################################################################