import tensorflow as tf

model = '../model/accelerate.pb'    #pb文件名称
self_module = tf.load_op_library('../StemConv.so')
self_module = tf.load_op_library('../Cell.so')
#model = '../model/model.pb'
with tf.Graph().as_default():
    graph_def = tf.GraphDef()
    with open(model, "rb") as f:
        graph_def.ParseFromString(f.read())
        _ = tf.import_graph_def(graph_def, name="")
    graph = tf.get_default_graph()
    num = 0
    for var in tf.trainable_variables():
        num += np.prod(var.get_shape().as_list())
        print('-----------------------------')
        print(var.get_shape().as_list())

    run_meta = tf.RunMetadata()
    #with graph.as_default():
    opts = tf.profiler.ProfileOptionBuilder.float_operation()
    res = tf.profiler.profile(graph, run_meta=run_meta, cmd='op', options=opts)
    flops = res.total_float_ops // 2
    print('flops is {}::{}K::{}M'.format(flops, flops//1000, flops//1000//1000))
    print('parameter num is {}::{}K::{:.1f}M'.format(num, num//1000, num//1000/1000))
