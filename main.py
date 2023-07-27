# Demo - train the DenseFuse network & use it to generate an image

from __future__ import print_function

from train_recons import train_recons
from generate import generate
from utils import list_images
import os
import re

os.environ["CUDA_VISIBLE_DEVICES"] = "0,1"

# True for training phase
IS_TRAINING = False
# True for video sequences(frames)
IS_VIDEO = False
# True for RGB images
IS_RGB = False

# True - 1000 images for validation
# This is a very time-consuming operation when TRUE
IS_Validation = False

BATCH_SIZE = 2
EPOCHES = 4

# MODEL_SAVE_PATH = './models/deepfuse_dense_model_bs4_epoch2_relu_pLoss_noconv_test.ckpt'
# model_pre_path  = './models/deepfuse_dense_model_bs2_epoch2_relu_pLoss_noconv_NEW.ckpt'

# In testing process, 'model_pre_path' is set to None
# The "model_pre_path" in "main.py" is just a pre-train model and not necessary for training and testing. 
# It is set as None when you want to train your own model. 
# If you already train a model, you can set it as your model for initialize weights.
model_pre_path = None

def main():

	if IS_TRAINING:
                ir_trn_imgs_path = list_images('/home/siting/Siting/datasets/LLVIP/infrared/train')
                vi_trn_imgs_path = list_images('/home/siting/Siting/datasets/LLVIP/visible/train')
                ir_val_imgs_path = list_images('/home/siting/Siting/datasets/LLVIP/infrared/val')
                vi_val_imgs_path = list_images('/home/siting/Siting/datasets/LLVIP/visible/val')

                ssim_weight = 10
                model_save_path = './models/model_0723_9k_qat.ckpt'
                
                print('\nBegin to train the network ...\n')
                
                train_recons(ir_trn_imgs_path, vi_trn_imgs_path, ir_val_imgs_path, vi_val_imgs_path,  model_save_path, model_pre_path, ssim_weight, EPOCHES, BATCH_SIZE, IS_Validation, debug=True)
                print('\nSuccessfully! Done training...\n')
	else:
		if IS_VIDEO:
			ssim_weight = SSIM_WEIGHTS[0]
			model_path = MODEL_SAVE_PATHS[0]

			IR_path = list_images('video/1_IR/')
			VIS_path = list_images('video/1_VIS/')
			output_save_path = 'video/fused'+ str(ssim_weight) +'/'
			generate(IR_path, VIS_path, model_path, model_pre_path,
			         ssim_weight, 0, IS_VIDEO, 'addition', output_path=output_save_path)
		else:

                    ssim_weight = 10
                    model_path = './models/model_0723_9k_qat.ckpt'
                    print('\nBegin to generate pictures ...\n')
		    # path = 'images/IV_images/'
                    infrared_path = list_images('/home/siting/Siting/datasets/LLVIP/infrared/val')
                    visible_path = list_images('/home/siting/Siting/datasets/LLVIP/visible/val')
                    
                    #infrared_path = list_images('/home/siting/Siting/datasets/nchu/dongshi_0000/ir')
                    #visible_path = list_images('/home/siting/Siting/datasets/nchu/dongshi_0000/vi')
                    
                    for i in range(len(infrared_path)):
                        index = re.findall(r"\d+", infrared_path[i])
                      
                      #print(infrared_path[i])
                        #print(index[0])
                        # RGB images
                        infrared = infrared_path[i]
                        visible = visible_path[i]

                        # choose fusion layer
                        fusion_type = 'QAT'
                        output_save_path = './outputs_test/' + index[0]
                        #output_save_path = './outputs_test/' + index[0]
                        #print(infrared)
                        generate(infrared, visible, model_path, model_pre_path,
		            ssim_weight, index[0], IS_VIDEO, IS_RGB, type = fusion_type, output_path = output_save_path)
                    


if __name__ == '__main__':
    main()

