import tensorflow as tf

outfile = './output_nodes.txt'

with tf.Session() as sess:
    tf.train.import_meta_graph('{model_name}.ckpt.meta', clear_devices=True)
    graph_def = tf.get_default_graph().as_graph_def(add_shapes=True)
    node_list = [n.name for n in graph_def.node]
    with open(outfile, "w") as f:
        for node in node_list:
            print("node_name", node)
            f.write(node + "\n")
