import tensorflow as tf
import argparse
import time
import numpy as np
from tensorflow.core.protobuf import rewriter_config_pb2

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--model_path', default='./model/model.pb',  type=str, help='model path')
    parser.add_argument('--input_node', default='images_train', type=str, help='input node name')
    parser.add_argument('--output_node', default='child/Relu', type=str, help='output node name')
    parser.add_argument('--loops', default=1000, type=str, help='the number of test loops')

    args = parser.parse_args()

    cpu_num = 1
    config = tf.ConfigProto(device_count={"CPU": cpu_num},
                inter_op_parallelism_threads = cpu_num,
                intra_op_parallelism_threads = cpu_num,
                log_device_placement=True)
    config.graph_options.rewrite_options.layout_optimizer=rewriter_config_pb2.RewriterConfig.OFF
    with tf.device('/cpu:0'):
        test_image = np.ones((1, 32, 32, 3))
        with tf.Graph().as_default():
            graph_def = tf.GraphDef()
            with open(args.model_path, "rb") as f:
                graph_def.ParseFromString(f.read())
            # fix nodes
            for node in graph_def.node:
                if node.op == 'RefSwitch':
                    node.op = 'Switch'
                    for index in xrange(len(node.input)):
                        if 'moving_' in node.input[index]:
                            node.input[index] = node.input[index] + '/read'
                elif node.op == 'AssignSub':
                    node.op = 'Sub'
                    if 'use_locking' in node.attr: del node.attr['use_locking']
                elif node.op == 'AssignAdd':
                    node.op = 'Add'
                    if 'use_locking' in node.attr: del node.attr['use_locking']
            _ = tf.import_graph_def(graph_def, name="")
            from IPython import embed
            #embed()
                
            with tf.Session(config=config) as sess:
                result = sess.run('{}:0'.format(args.output_node), feed_dict={'images_train:0':test_image})
                print(result)
                t = 0
                for i in range(args.loops):
                    t1 = time.time()
                    sess.run('{}:0'.format(args.input_node), feed_dict={'images_train:0':test_image})
                    t2 = time.time()
                    sess.run('{}:0'.format(args.output_node), feed_dict={'images_train:0':test_image})
                    t3 = time.time()
                    t += t3 - t2 - t2 + t1
            print ('Total time of %d loops: %f seconds, each %f ms' % (args.loops, t, t * 1000 / args.loops))
    
