#coding=utf-8
# 本题为考试多行输入输出规范示例，无需提交，不计分。
import sys
import math
if __name__ == "__main__":
    # 读取第一行的n
    l = sys.stdin.readline().strip()
    l=list(map(int, l.split()))
    m=int(l[0])
    n=int(l[1])
    line=[]
    for i in range(n):
        line.append(int(input()))

    from keras.models import load_model

    import tensorflow as tf

    import os

    import os.path as osp

    from keras import backend as K

    # 路径参数

    input_path = './model_data/'

    weight_file = 'yolo.h5'

    weight_file_path = osp.join(input_path, weight_file)

    output_graph_name = weight_file[:-3] + '.pb'


    # 转换函数

    def h5_to_pb(h5_model, output_dir, model_name, out_prefix="output_", log_tensorboard=True):

        if osp.exists(output_dir) == False:
            os.mkdir(output_dir)

        out_nodes = []

        for i in range(len(h5_model.outputs)):
            out_nodes.append(out_prefix + str(i + 1))

            tf.identity(h5_model.output[i], out_prefix + str(i + 1))

        sess = K.get_session()

        from tensorflow.python.framework import graph_util, graph_io

        init_graph = sess.graph.as_graph_def()

        main_graph = graph_util.convert_variables_to_constants(sess, init_graph, out_nodes)

        graph_io.write_graph(main_graph, output_dir, name=model_name, as_text=False)

        if log_tensorboard:
            from tensorflow.python.tools import import_pb_to_tensorboard

            import_pb_to_tensorboard.import_to_tensorboard(osp.join(output_dir, model_name), output_dir)


    # 输出路径

    output_dir = osp.join(os.getcwd(), "trans_model")

    # 加载模型

    h5_model = load_model(weight_file_path)

    h5_to_pb(h5_model, output_dir=output_dir, model_name=output_graph_name)

    print('model saved')
    line=sorted(line)
    ans=1
    for i, j in zip(line, line[1:]):
        ans+=j//i-1
        # print('///',j//i)
        # if j/i-j//i==0 and j//i==0:
        #     ans-=1
    if line[0]==1:
        if len(line)!=1:
            print(ans)
        else:
            print(1)
    else:
        print(-1)