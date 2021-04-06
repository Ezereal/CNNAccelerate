import tensorflow as tf

model = '../model/model.pb'    #pb文件名称
self_module = tf.load_op_library('../StemConv.so')
self_module = tf.load_op_library('../Cell.so')
#model = '../model/model.pb'
graph = tf.get_default_graph()
graph_def = graph.as_graph_def()
graph_def.ParseFromString(tf.gfile.FastGFile(model, 'rb').read())
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
tf.import_graph_def(graph_def, name='graph')
summaryWriter = tf.summary.FileWriter('../log/', graph)   #log存放地址
