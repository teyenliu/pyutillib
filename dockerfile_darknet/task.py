from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

from absl import flags
import os
import sys
import tensorflow as tf

flags.DEFINE_string(
    'data_config', None, 'Path to data config file, exp:cfg/voc.data')
flags.DEFINE_string(
    'model_config', None, 'Path to model  config file, exp:cfg/yolov3-voc.cfg')
flags.DEFINE_string(
    'pretrain_file', "", 'Path to directory holding a pretrain weights and biases.')
flags.DEFINE_string(
    'gpus_list', "0", 'specify the gpu number wanted to use in list, exp:0,1,2,3')
	
FLAGS = flags.FLAGS


def main(unused_argv):
  flags.mark_flag_as_required('model_config')
  flags.mark_flag_as_required('data_config')

  task_command = "darknet/darknet detector train " + FLAGS.data_config + \
        " " + FLAGS.model_config + \
        " " + FLAGS.pretrain_file + \
        " -gpus " + FLAGS.gpus_list

  print(task_command)
  os.system(task_command)


if __name__ == '__main__':
  tf.app.run()

