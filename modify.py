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
def init_weights():
    weight_names = []
    name = 'child/stem_conv/w/read'
    weight_names.append(name)
    name = 'child/stem_conv/bn/scale/read'
    weight_names.append(name)
    name = 'child/stem_conv/bn/offset/read'
    weight_names.append(name)
    return weight_names


if __name__ == '__main__':

    parser = argparse.ArgumentParser()
    parser.add_argument('--model_path', default='./model/model.pb',  type=str, help='model path')
    parser.add_argument('--output_path', default='./model.pb',  type=str, help='modify model path')
    parser.add_argument('--input_node', default='images_train', type=str, help='input node name')
    parser.add_argument('--output_node', default='child/stem_conv/bn/Identity', type=str, help='output node name')

    args = parser.parse_args()

    pb_file = args.model_path
    new_pb_file = args.output_path

    graph_def = tf.GraphDef()
    with tf.gfile.GFile(pb_file, "rb") as g:
        graph_def.ParseFromString(g.read())

    input_node = args.input_node
    output_node = args.output_node

    weight_names = init_weights()
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

with tf.Session() as sess:
    #converted_graph_def = tf.graph_util.convert_variables_to_constants(sess, graph_def, ['loss/Softmax'])
    converted_graph_def = graph_def
    tf.train.write_graph(converted_graph_def, './', new_pb_file, as_text=False)

