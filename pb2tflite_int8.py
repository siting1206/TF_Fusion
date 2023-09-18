import tensorflow as tf

path_to_frozen_graphdef_pb = 'qat_9k_relu/pb_model/freeze_eval_graph.pb'
input_shapes = {'content':[1,None,None,1], 'style':[1,None,None,1]}
#input_shapes = {'content':[1,640,480,1], 'style':[1,640,480,1]}
input_arrays = ["content", "style"]
output_arrays = ["BiasAdd_10"]


# int8(int8 when inference)
converter = tf.lite.TFLiteConverter.from_frozen_graph(path_to_frozen_graphdef_pb,
        input_arrays, output_arrays)
converter.target_ops = [tf.lite.OpsSet.TFLITE_BUILTINS,tf.lite.OpsSet.SELECT_TF_OPS]
#converter.target_ops = [tf.lite.OpsSet.TFLITE_BUILTINS_INT8]
converter.inference_type = tf.uint8
converter.quantized_input_stats = {'content':(0., 1.), 'style':(0., 1.)}  # mean, std_dev
converter.default_ranges_stats = (0, 255)
converter.inference_type = tf.uint8

tflite_uint8_model = converter.convert()
open("model_9k_relu.tflite", "wb").write(tflite_uint8_model)
