"""
Generate synthetic image/seg pairs on the fly 
"""
import time
from options.train_options import TrainOptions
from data import create_dataset
from models import create_model
import os
import matplotlib.pyplot as plt

if __name__ == '__main__':
    opt = TrainOptions().parse()   # get training options

    dataset = create_dataset(opt)  # create a dataset given opt.dataset_mode and other options
    dataset_size = len(dataset)    # get the number of images in the dataset.
    print('The number of training images = %d' % dataset_size)

    # hard-code cyclegan related parameters
    opt.model = 'cycle_gan'
    model = create_model(opt)      # create a cyclegan model
    model.isTrain = False
    model.setup(opt)               # regular setup: load and print networks; create schedulers
    model.eval()

    total_iters = 0                # the total number of training iterations
    os.makedirs('./train_images/%s'%opt.name, exist_ok=True)
    for epoch in range(opt.epoch_count, opt.n_epochs + opt.n_epochs_decay + 1):    # outer loop for different epochs; we save the model by <epoch_count>, <epoch_count>+<save_latest_freq>
        epoch_start_time = time.time()  # timer for entire epoch
        iter_data_time = time.time()    # timer for data loading per iteration
        epoch_iter = 0                  # the number of training iterations in current epoch, reset to 0 every epoch
        for i, data in enumerate(dataset):  # inner loop within one epoch
            total_iters += opt.batch_size
            epoch_iter += opt.batch_size
            # generate training image
            model.set_input(data)         # unpack data from dataset and apply preprocessing
            # run inference
            model.test()
            visuals = model.get_current_visuals()

            input_image = visuals['fake_B']
            target_mask = visuals['real_A']

            '''sanity check'''
            plt.figure(figsize=(10,5))
            for fig_i, img_i in enumerate([input_image, target_mask]):
                image = img_i[0].permute(1,2,0).detach().cpu().numpy()
                image = (image + 1.0) / 2.0
                plt.subplot(1, 2, fig_i+1)
                plt.imshow(image)
                plt.axis('off')
            plt.savefig('./train_images/%s/ep%d_%d.jpg'%(opt.name, epoch, i), dpi=200)
            plt.close()

            '''add the segmentation training below
            pred = net(input_image)
            loss = fn(pred, target_mask)
            ...
            '''



        print('End of epoch %d / %d \t Time Taken: %d sec' % (epoch, opt.n_epochs + opt.n_epochs_decay, time.time() - epoch_start_time))
