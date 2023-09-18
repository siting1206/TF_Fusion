# Train the DenseFuse Net

from __future__ import print_function

import scipy.io as scio
import numpy as np
import tensorflow as tf

from ssim_loss_function import SSIM_LOSS
from l1_loss import L1_LOSS

from densefuse_net import DenseFuseNet
from utils import get_train_images, get_train_images_rgb
import os
os.environ["CUDA_VISIBLE_DEVICES"]="0,1"
print("gpu: ", tf.test.is_gpu_available())

STYLE_LAYERS = ('relu1_1', 'relu2_1', 'relu3_1', 'relu4_1')

HEIGHT = 256
WIDTH = 256
CHANNELS = 1 # gray scale, default

LEARNING_RATE = 1e-4
EPSILON = 1e-5


def gray_sobel(inputs):
    assert inputs.shape[3]==1
    filter_x = tf.constant([[-1,0,1],
                            [-2,0,2],
                            [-1,0,1]], tf.float32)
    filter_x = tf.reshape(filter_x, [3,3,1,1])
    filter_y = tf.transpose(filter_x, [1,0,2,3])
    res_x = tf.nn.conv2d(inputs, filter_x, strides=[1,1,1,1], padding='SAME')
    res_y = tf.nn.conv2d(inputs, filter_y, strides=[1,1,1,1], padding='SAME')
    res_xy = tf.abs(res_x) + tf.abs(res_y)
    return res_xy

def train_recons(ir_trn_imgs_path, vi_trn_imgs_path, ir_val_imgs_path, vi_val_imgs_path, save_path, model_pre_path, ssim_weight, EPOCHES_set, BATCH_SIZE, IS_Validation, debug=False, logging_period=1):
    if debug:
        from datetime import datetime
        start_time = datetime.now()
    EPOCHS = EPOCHES_set
    print("EPOCHES   : ", EPOCHS)
    print("BATCH_SIZE: ", BATCH_SIZE)

    num_imgs = len(ir_trn_imgs_path)
    
    ir_trn_imgs_path = ir_trn_imgs_path[:num_imgs]
    vi_trn_imgs_path = vi_trn_imgs_path[:num_imgs]
    mod = num_imgs % BATCH_SIZE

    print('Train images number %d.\n' % num_imgs)
    print('Train images samples %s.\n' % str(num_imgs / BATCH_SIZE))

    if mod > 0:
        print('Train set has been trimmed %d samples...\n' % mod)
        ir_trn_imgs_path = ir_trn_imgs_path[:-mod]
        vi_trn_imgs_path = vi_trn_imgs_path[:-mod]

    # get the traing image shape
    INPUT_SHAPE_OR = (BATCH_SIZE, HEIGHT, WIDTH, CHANNELS)

        
    moodel_save_path = './models/model_0827_9k_qat_add.ckpt'
    

    # create the graph
    with tf.Graph().as_default(), tf.Session() as sess:
        ir = tf.placeholder(tf.float32, shape=INPUT_SHAPE_OR, name='ir')
        vi = tf.placeholder(tf.float32, shape=INPUT_SHAPE_OR, name='vi')
        ir_source = ir
        vi_source = vi

        # print('source  :', source.shape)
        print('original: ', ir_source.shape)
        print('style: ', vi_source.shape)

        # create the deepfuse net (encoder and decoder)
        dfn = DenseFuseNet(model_pre_path)
        fusion = dfn.transform_addition(ir_source, vi_source)
        print('fusion:', fusion.shape)

        # sobel
        ir_grad = gray_sobel(ir)
        vi_grad = gray_sobel(vi)
        fusion_grad = gray_sobel(fusion)

        max_grad = tf.maximum(ir_grad, vi_grad)
        l_grad = L1_LOSS(fusion_grad, max_grad)

        l_int = (20*L1_LOSS(fusion, vi) + (1-SSIM_LOSS(fusion, vi)))*0.5 + (20*L1_LOSS(fusion, ir) + (1-SSIM_LOSS(fusion, ir)))*0.5

        loss = l_int + 20*l_grad

        ## quantization aware training
        g = tf.get_default_graph()
        tf.contrib.quantize.create_training_graph(input_graph=g, quant_delay=3000000)


        train_op = tf.train.AdamOptimizer(LEARNING_RATE).minimize(loss)

        sess.run(tf.global_variables_initializer())

        # saver = tf.train.Saver()
        saver = tf.train.Saver(keep_checkpoint_every_n_hours=1)

        # ** Start Training **
        step = 0
        count_loss = 0
        n_batches = int(len(ir_trn_imgs_path) // BATCH_SIZE)
        val_batches = int(len(ir_val_imgs_path) // BATCH_SIZE)

        if debug:
            elapsed_time = datetime.now() - start_time
            print('\nElapsed time for preprocessing before actually train the model: %s' % elapsed_time)
            print('Now begin to train the model...\n')
            start_time = datetime.now()

        Loss_all = [i for i in range(EPOCHS * n_batches)]
        Loss_grad = [i for i in range(EPOCHS * n_batches)]
        Loss_int = [i for i in range(EPOCHS * n_batches)]
        Val_grad_data = [i for i in range(EPOCHS * n_batches)]
        Val_int_data = [i for i in range(EPOCHS * n_batches)]
        for epoch in range(EPOCHS):

            seed = list(zip(ir_trn_imgs_path, vi_trn_imgs_path))
            np.random.shuffle(seed)

            ir_trn_imgs_path = [s[0] for s in seed]
            vi_trn_imgs_path = [s[1] for s in seed]

            for batch in range(n_batches):
                # retrive a batch of content and style images

                ir_trn_path = ir_trn_imgs_path[batch*BATCH_SIZE:(batch*BATCH_SIZE + BATCH_SIZE)]
                vi_trn_path = vi_trn_imgs_path[batch*BATCH_SIZE:(batch*BATCH_SIZE + BATCH_SIZE)]
                ### read gray scale images
                ir_trn_batch = get_train_images(ir_trn_path, crop_height=HEIGHT, crop_width=WIDTH, flag=False)
                vi_trn_batch = get_train_images(vi_trn_path, crop_height=HEIGHT, crop_width=WIDTH, flag=False)
                ### read RGB images
                # original_batch = get_train_images_rgb(original_path, crop_height=HEIGHT, crop_width=WIDTH, flag=False)
                #print(ir_trn_batch.shape)
                ir_trn_batch = ir_trn_batch.transpose((3, 0, 1, 2))
                vi_trn_batch = vi_trn_batch.transpose((3, 0, 1, 2))


                # print('original_batch shape final:', original_batch.shape)
            
                # run the training step
                sess.run(train_op, feed_dict={ir: ir_trn_batch, vi: vi_trn_batch})
                
                step += 1
                if debug:
                    is_last_step = (epoch == EPOCHS - 1) and (batch == n_batches - 1)

                    if is_last_step or step % logging_period == 0:
                        elapsed_time = datetime.now() - start_time
                        _l_grad, _loss, _l_int = sess.run([l_grad, loss, l_int], feed_dict={ir: ir_trn_batch, vi: vi_trn_batch})
                        Loss_all[count_loss] = _loss
                        Loss_grad[count_loss] = _l_grad
                        Loss_int[count_loss] = _l_int
                        print('epoch: %d/%d, step: %d,  total loss: %s, elapsed time: %s' % (epoch, EPOCHS, step, _loss, elapsed_time))
                        print('l_grad_loss: %s, l_int_loss: %s ,w_ssim_loss: %s ' % (_l_grad, _l_int, ssim_weight * _l_int))

                        # IS_Validation = True;
                        # Calculating the accuracy rate for 1000 images, every 100 steps
                        if IS_Validation:
                            val_grad_acc = 0
                            val_int_acc = 0
                            seed_val = list(zip(ir_val_imgs_path, vi_val_imgs_path))
                            np.random.shuffle(seed_val)
                            ir_val_imgs_path = [s[0] for s in seed_val]
                            vi_val_imgs_path = [s[1] for s in seed_val]
                            #np.random.shuffle(validatioin_imgs_path)
                            val_start_time = datetime.now()
                            for v in range(val_batches):
                                ir_val_path = ir_val_imgs_path[v * BATCH_SIZE:(v * BATCH_SIZE + BATCH_SIZE)]
                                vi_val_path = vi_val_imgs_path[v * BATCH_SIZE:(v * BATCH_SIZE + BATCH_SIZE)]

                                ir_val_batch = get_train_images(ir_val_path, crop_height=HEIGHT, crop_width=WIDTH,flag=False)
                                vi_val_batch = get_train_images(vi_val_path, crop_height=HEIGHT, crop_width=WIDTH,flag=False)

                                ir_val_batch = ir_val_batch.reshape([BATCH_SIZE, 256, 256, 1])
                                vi_val_batch = vi_val_batch.reshape([BATCH_SIZE, 256, 256, 1])
                                val_grad, val_int = sess.run([l_grad, l_int], feed_dict={ir: ir_val_batch, vi: vi_val_batch})
                                val_grad_acc = val_grad_acc + val_grad
                                val_int_acc = val_int_acc + val_int
                            Val_grad_data[count_loss] = val_grad_acc/val_batches
                            Val_int_data[count_loss] = val_int_acc / val_batches
                            val_es_time = datetime.now() - val_start_time
                            print('validation value, l_grad: %s, l_int: %s, elapsed time: %s' % (val_grad_acc/val_batches, val_int_acc / val_batches, val_es_time))
                            print('------------------------------------------------------------------------------')
                        count_loss += 1
                

        # ** Done Training & Save the model **
        #g = tf.get_default_graph()
        #tf.contrib.quantize.create_eval_graph(input_graph=g)


        saver.save(sess, save_path)

        loss_data = Loss_all[:count_loss]
        scio.savemat('./models/loss/DeepDenseLossData_0711'+str(ssim_weight)+'.mat',{'loss':loss_data})

        loss_grad_data = Loss_grad[:count_loss]
        scio.savemat('./models/loss/DeepDenseLossGradDat_0711'+str(ssim_weight)+'.mat', {'loss_grad': loss_grad_data})

        loss_int_data = Loss_int[:count_loss]
        scio.savemat('./models/loss/DeepDenseLossIntData_0711.mat'+str(ssim_weight)+'', {'loss_int': loss_int_data})

        # IS_Validation = True;
        if IS_Validation:
            validation_ssim_data = Val_ssim_data[:count_loss]
            scio.savemat('./models/val/Validation_grad_Data.mat' + str(ssim_weight) + '', {'val_grad': validation_ssim_data})
            validation_pixel_data = Val_pixel_data[:count_loss]
            scio.savemat('./models/val/Validation_int_Data.mat' + str(ssim_weight) + '', {'val_int': validation_pixel_data})


        if debug:
            elapsed_time = datetime.now() - start_time
            print('Done training! Elapsed time: %s' % elapsed_time)
            print('Model is saved to: %s' % save_path)

