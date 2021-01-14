import tensorflow as tf
from tensorflow.core.framework import attr_value_pb2
from tensorflow.core.framework import graph_pb2
from tensorflow.core.framework import node_def_pb2
from tensorflow.python.framework import dtypes
from tensorflow.python.framework import tensor_shape
from tensorflow.python.framework import tensor_util
import sys
import argparse

def set_attr_shape(node, key, value):
  try:
    node.attr[key].CopyFrom(
        attr_value_pb2.AttrValue(shape=tensor_shape.as_shape(value).as_proto()))
  except KeyError:
    pass

def set_attr_tensor(node, key, value, dtype, shape=None):
  try:
    node.attr[key].CopyFrom(
        attr_value_pb2.AttrValue(tensor=tensor_util.make_tensor_proto(
            value, dtype=dtype, shape=shape)))
  except KeyError:
    pass

def create_node(op, name, inputs):
    new_node = node_def_pb2.NodeDef()
    new_node.op = op
    new_node.name = name
    for input_name in inputs:
        new_node.input.extend([input_name])
    return new_node

def set_attr_dtype(node, key, value):
  try:
    node.attr[key].CopyFrom(
        attr_value_pb2.AttrValue(type=value.as_datatype_enum))
  except KeyError:
    pass

def create_constant_node(name, value, dtype, shape=None):
    node = create_node("Const", name, [])
    set_attr_dtype(node, "dtype", dtype)
    set_attr_tensor(node, "value", value, dtype, shape)
    return node

def create_placeholder(name):
    node = create_node('Placeholder', name, [])
    set_attr_dtype(node, "dtype", dtypes.int32)
    set_attr_shape(node, "shape", [None, 128])
    return node

################################################################################
def init_stem_weights():
    weight_names = []
    name = 'child/stem_conv/w/read'
    weight_names.append(name)
    name = 'child/stem_conv/bn/scale/read'
    weight_names.append(name)
    name = 'child/stem_conv/bn/offset/read'
    weight_names.append(name)
    return weight_names

def init_cell_weights():
    weight_names = []
    # pool_x
    weight_names.append('child/layer_0/calibrate/pool_x/w/read')
    weight_names.append('child/layer_0/calibrate/pool_x/bn/scale/read')
    weight_names.append('child/layer_0/calibrate/pool_x/bn/offset/read')
    # poll_y
    weight_names.append('child/layer_0/calibrate/pool_y/w/read')
    weight_names.append('child/layer_0/calibrate/pool_y/bn/scale/read')
    weight_names.append('child/layer_0/calibrate/pool_y/bn/offset/read')
    # layer_base
    weight_names.append('child/layer_0/layer_base/w/read')
    weight_names.append('child/layer_0/layer_base/bn/scale/read')
    weight_names.append('child/layer_0/layer_base/bn/offset/read')
    # x_conv sep_conv_0
    weight_names.append('child/layer_0/cell_0/x_conv/sep_conv_0/w_depth/read')
    weight_names.append('child/layer_0/cell_0/x_conv/sep_conv_0/w_point/read')
    weight_names.append('child/layer_0/cell_0/x_conv/sep_conv_0/bn/scale/read')
    weight_names.append('child/layer_0/cell_0/x_conv/sep_conv_0/bn/offset/read')
    # x_conv sep_conv_1
    weight_names.append('child/layer_0/cell_0/x_conv/sep_conv_1/w_depth/read')
    weight_names.append('child/layer_0/cell_0/x_conv/sep_conv_1/w_point/read')
    weight_names.append('child/layer_0/cell_0/x_conv/sep_conv_1/bn/scale/read')
    weight_names.append('child/layer_0/cell_0/x_conv/sep_conv_1/bn/offset/read')
    # y_conv sep_conv_0
    weight_names.append('child/layer_0/cell_0/y_conv/sep_conv_0/w_depth/read')
    weight_names.append('child/layer_0/cell_0/y_conv/sep_conv_0/w_point/read')
    weight_names.append('child/layer_0/cell_0/y_conv/sep_conv_0/bn/scale/read')
    weight_names.append('child/layer_0/cell_0/y_conv/sep_conv_0/bn/offset/read')
    # y_conv sep_conv_1
    weight_names.append('child/layer_0/cell_0/y_conv/sep_conv_1/w_depth/read')
    weight_names.append('child/layer_0/cell_0/y_conv/sep_conv_1/w_point/read')
    weight_names.append('child/layer_0/cell_0/y_conv/sep_conv_1/bn/scale/read')
    weight_names.append('child/layer_0/cell_0/y_conv/sep_conv_1/bn/offset/read')
    # final_combine
    weight_names.append('child/layer_0/final_combine/calibrate_0/path1_conv/w/read')
    weight_names.append('child/layer_0/final_combine/calibrate_0/path2_conv/w/read')
    weight_names.append('child/layer_0/final_combine/calibrate_0/bn/scale/read')
    weight_names.append('child/layer_0/final_combine/calibrate_0/bn/offset/read')
    return weight_names


def modify_stem(graph_def):
    input_node = 'images_train'
    output_node = 'child/stem_conv/bn/Identity'

    weight_names = init_stem_weights()
    # Create the super Bert node
    input_nodes = [input_node]
    input_nodes.extend(weight_names)
    modify_node = create_node('StemConv', 'fused_stem', input_nodes)
    #set_attr_dtype(bert_node, "T", tf.int32) # hard code, may need to change
    graph_def.node.extend([modify_node])

    for node in graph_def.node:
        if not node.input:
            continue
        for i in range(len(node.input)):
            if str(node.input[i]) == output_node:
                node.input[i] = modify_node.name
                print('**** Modified the input node of %s' % node.name)
                break

def modify_cell(graph_def):
    input_node = 'child/stem_conv/bn/Identity'
    output_node = 'child/Mean'

    weight_names = init_cell_weights()
    # Create the super Bert node
    input_nodes = [input_node]
    input_nodes.extend(weight_names)
    modify_node = create_node('Cell', 'fused_cell', input_nodes)
    #set_attr_dtype(bert_node, "T", tf.int32) # hard code, may need to change
    graph_def.node.extend([modify_node])

    for node in graph_def.node:
        if not node.input:
            continue
        for i in range(len(node.input)):
            if str(node.input[i]) == output_node:
                node.input[i] = modify_node.name
                print('**** Modified the input node of %s' % node.name)
                break

if __name__ == '__main__':

    parser = argparse.ArgumentParser()
    parser.add_argument('--model_path', default='./model/model.pb',  type=str, help='model path')
    parser.add_argument('--output_path', default='./model.pb',  type=str, help='modify model path')

    args = parser.parse_args()

    pb_file = args.model_path
    new_pb_file = args.output_path

    graph_def = tf.GraphDef()
    with tf.gfile.GFile(pb_file, "rb") as g:
        graph_def.ParseFromString(g.read())

    modify_stem(graph_def)
    modify_cell(graph_def)


    with tf.Session() as sess:
        #converted_graph_def = tf.graph_util.convert_variables_to_constants(sess, graph_def, ['loss/Softmax'])
        converted_graph_def = graph_def
        tf.train.write_graph(converted_graph_def, './', new_pb_file, as_text=False)

