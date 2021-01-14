import tensorflow as tf
import numpy as np
import json
import time
import argparse
from tensorflow.core.protobuf import rewriter_config_pb2


if __name__ == '__main__':

    parser = argparse.ArgumentParser()
    parser.add_argument('--model_path', default='./model.pb',  type=str, help='modify model path')
    parser.add_argument('--input_node', default='child/stem_conv/bn/Identity', type=str, help='input node name')
    #parser.add_argument('--output_node', default='fused_cell', type=str, help='output node name')
    parser.add_argument('--output_node', default='child/fc/fc_result', type=str, help='output node name')
    parser.add_argument('--loops', default=1000, type=str, help='the number of test loops')

    args = parser.parse_args()

    self_module = tf.load_op_library('./StemConv.so')
    self_module = tf.load_op_library('./Cell.so')

    cpu_num = 1
    config = tf.ConfigProto(device_count={"CPU": cpu_num},
                inter_op_parallelism_threads = cpu_num,
                intra_op_parallelism_threads = cpu_num,
                log_device_placement=True)
    config.graph_options.rewrite_options.layout_optimizer=rewriter_config_pb2.RewriterConfig.OFF
    with tf.Session(config=config) as sess:
        g = tf.Graph().as_default()

        pb_file = args.model_path
        with open(pb_file, "rb") as f:
            g_def = tf.GraphDef()
            g_def.ParseFromString(f.read())
            _ = tf.import_graph_def(g_def, name="")

        test_image = np.ones((1, 32, 32, 3))

        # Warm up
        result = sess.run('{}:0'.format(args.output_node), feed_dict={'images_train:0':test_image})
        print (result)

        t = 0
        num = 0
        for i in range(args.loops):
            t1 = time.time()
            sess.run('{}:0'.format(args.input_node), feed_dict={'images_train:0':test_image})
            t2 = time.time()
            sess.run('{}:0'.format(args.output_node), feed_dict={'images_train:0':test_image})
            t3 = time.time()
            #if t3 - t2 - t2 + t1 > 0.002:
            #    continue
            num+=1
            t += t3 - t2 - t2 + t1
            #print((t3 - t2 - t2 + t1)*1000)

        print ('Total time of %d loops: %f seconds, each %f ms' % (args.loops, t, t * 1000 / num))
    


