import tensorflow as tf
import numpy as np
import re
import os

from PIL import Image
from utils import list_images, save_images

infrared_paths = list_images('/home/datasets/nchu/dongshi_0000/ir')
visible_paths = list_images('/home/datasets/nchu/dongshi_0000/vi')

for i in range(len(infrared_paths)):
    index = re.findall(r"\d+", infrared_paths[i])
    infrared_path = infrared_paths[i]
    visible_path = visible_paths[i]

    path = './outputs_test'
    if not os.path.isdir(path):
            os.mkdir(path)
    output_save_path = './outputs_test/' + index[2]

    ir = Image.open(infrared_path)
    ir = ir.convert('L')
    ir = ir.resize((640, 480))
    ir = np.array(ir, np.uint8)

    vi = Image.open(visible_path)
    vi = vi.convert('L')
    vi = vi.resize((640, 480))
    vi = np.array(vi, np.uint8)

    dimension = ir.shape
    ir = ir.reshape([1, dimension[0], dimension[1], 1])
    vi = vi.reshape([1, dimension[0], dimension[1], 1])

    # Load TFLite model and allocate tensors.
    interpreter = tf.lite.Interpreter(model_path='./nchu_uint8.tflite')
    interpreter.allocate_tensors()

    # Get input and output tensors.
    input_details = interpreter.get_input_details()
    output_details = interpreter.get_output_details()

    interpreter.set_tensor(input_details[0]['index'], ir)
    interpreter.set_tensor(input_details[1]['index'], vi)
    interpreter.invoke()
    output_data = interpreter.get_tensor(output_details[0]['index'])
    save_images(infrared_paths[i], output_data, output_save_path, suffix='.jpg')

