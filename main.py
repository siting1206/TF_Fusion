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

model_pre_path = None

def main():

	if IS_TRAINING:
                ir_trn_imgs_path = list_images('/home/siting/Siting/datasets/LLVIP/infrared/train')
                vi_trn_imgs_path = list_images('/home/siting/Siting/datasets/LLVIP/visible/train')
                ir_val_imgs_path = list_images('/home/siting/Siting/datasets/LLVIP/infrared/val')
                vi_val_imgs_path = list_images('/home/siting/Siting/datasets/LLVIP/visible/val')

                ssim_weight = 10
                model_save_path = './models/model_9k_relu_qat.ckpt'
                
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
                    model_path = './models/model_9k_relu_qat.ckpt'
                    print('\nBegin to generate pictures ...\n')
                    infrared_path = list_images('/home/siting/Siting/datasets/LLVIP/infrared/val')
                    visible_path = list_images('/home/siting/Siting/datasets/LLVIP/visible/val')
                    
                    #infrared_path = list_images('/home/siting/Siting/datasets/nchu/dongshi_0000/ir')
                    #visible_path = list_images('/home/siting/Siting/datasets/nchu/dongshi_0000/vi')
                    
                    #infrared_path = list_images('/home/siting/Siting/datasets/soldier/ir')
                    #visible_path = list_images('/home/siting/Siting/datasets/soldier/vi')
                    
                    for i in range(len(infrared_path)):
                        index = re.findall(r"\d+", infrared_path[i])
                      
                        infrared = infrared_path[i]
                        visible = visible_path[i]

                        # choose fusion layer
                        fusion_type = 'qat_test'
                        output_save_path = './9k_ReLU_qat/' + index[0]
                        generate(infrared, visible, model_path, model_pre_path,
		            ssim_weight, index[0], IS_VIDEO, IS_RGB, type = fusion_type, output_path = output_save_path)

if __name__ == '__main__':
    main()

