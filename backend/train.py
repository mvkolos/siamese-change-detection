from aeronet.backend.losses import bce_jaccard_loss
from aeronet.backend.metrics import iou_score, f1_score

from keras.models import load_model
from keras.optimizers import Adam
from keras.callbacks import TensorBoard, ModelCheckpoint
from torch.utils.data.sampler import RandomSampler

from backend.models import cd_unet
from backend.dataset import SeriesDataset
from backend.generator import segmentation_generator
from backend.utils import pad_x32, load_bc
from backend.augmentations import train_augmenter

import pandas as pd
import os
import argparse

test_dir = '../../data/california/ventura_test'
train_samples_dir = '../../data/california/samples/train_512'

channels_pre = ['RED', 'GRN', 'BLU']
channels_post = ['PRED', 'PGRN', 'PBLU']
output_labels = ['801']
num_samples = 1000
sample_size = (352, 352)
batch_size = 8
lr = 1e-4
train_steps = 300
validation_steps = 200
epochs = 20
start = 0

def run_training(fraction, mode, aug, n_epochs):
    experiment_name = f'{mode}_{fraction}_{int(aug)}_os'
    
    # model
    if mode=='imagenet':
        model = cd_unet(backbone_name = 'resnet18')
    elif mode=='random':
        model = cd_unet(backbone_name = 'resnet18', encoder_weights = None)
    elif mode=='tune':
        model_dir = args.model_dir
        model = load_model(f'{model_dir}/model.h5', compile=False)
        
        for layer in model.layers:
            layer.trainable = True 
        print('unfreezing: ', model.layers[2].trainable)
        base_dir = model_dir.split('/')[-1]
        experiment_name = f'TUNE_{base_dir}_{fraction}_os'
        
    # train data
    train_df = pd.read_csv('metadata/train.csv')
    n = train_df.shape[0]//fraction

    train_frames = train_df.frame.values[:n]
    print('mode: ', mode)
    print('train frames: ', train_frames, ' total: ', len(train_frames))
    
    band_collections  = {}

    for frame in train_frames:
        band_collections[frame] = load_bc(f'{train_samples_dir}/{frame}', 'ventura', \
                                          channels_pre+channels_post, output_labels)
    bc_test =  load_bc(test_dir, 'ventura_test', channels_pre+channels_post, output_labels)
    sampler = RandomSampler(range(n), replacement = True, num_samples = num_samples)
    
    transform = None
    if aug:
        transform = train_augmenter
        
    print('transform: ', transform)

    train_dataset = SeriesDataset(list(band_collections.values()), sample_size, channels_pre, \
                                  channels_post, output_labels, transform=transform)
    
    train_gen = segmentation_generator(train_dataset, sampler=sampler, batch_size = batch_size, num_workers=2)
    
    # validation data
    pre = bc_test.ordered(*channels_pre).numpy().transpose(1,2,0)
    pre, pad = pad_x32(pre)
    pre = pre[None,:,:,:]

    post = bc_test.ordered(*channels_post).numpy().transpose(1,2,0)
    post, pad = pad_x32(post)
    post = post[None,:,:,:]
    
    mask = bc_test.ordered(*output_labels).numpy()
    mask, pad = pad_x32(mask.squeeze()[:,:,None])
    mask = mask[None,:,:,:]

    validation_data = ([pre, post], mask)
    
    # experiment setup 
    # logging     
    log_dir = f'logs_4/{experiment_name}'
    model_dir = f'checkpoints_4/{experiment_name}'

    os.makedirs(log_dir, exist_ok=True)
    os.makedirs(model_dir, exist_ok=True)

    tensorboard = TensorBoard(log_dir=log_dir)
    model_checkpoint = ModelCheckpoint(f'{model_dir}/model.h5', save_best_only=True, \
                                       monitor = 'val_iou_score', mode='max', verbose=1)


    optimizer = Adam(lr = lr)
    loss = bce_jaccard_loss
    metrics = [iou_score, f1_score]
    model.compile(optimizer = optimizer, loss=loss, metrics = metrics)
    


    model.fit_generator(train_gen, epochs = start+ n_epochs, steps_per_epoch = train_steps, \
                    validation_data=validation_data, \
                    validation_steps = 1, callbacks = [tensorboard, model_checkpoint], \
                    initial_epoch=start)

def main(args):
    fractions = [int(f) for f in args.fraction.split()]
    mode = args.mode
    aug = args.aug
    n_epochs = args.n_epochs
    
    for fraction in fractions:
        run_training(fraction, mode, aug, n_epochs)
    

        
if __name__ == '__main__':

    parser = argparse.ArgumentParser()
    parser.add_argument('--fraction', type=str, default='1 2 4 8 16')
    parser.add_argument('--mode', type=str, default='imagenet')
    parser.add_argument('--model_dir', type=str, default='')    
    parser.add_argument('--aug', type=bool, default=False)
    parser.add_argument('--n_epochs', type=int, default=20)
    args = parser.parse_args()

    main(args)
    print('Exiting...')
    os._exit(0)