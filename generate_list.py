# -*- coding: utf-8 -*-

"""
generate list.txt
"""

"""
Example Usage:
-------------------
python generate_list.py \
    --data_dir: Path to the main dir. (directory).
    --output_path: Path to .txt.
    
The file tree:
    --cat_and_dog (the main dir)
        --cat
            --cat_0.jpg
            --cat_1.jpg
            --...
        --dog
            --dog_0.jpg
            --dog_1.jpg
            --...
"""

import os
import tensorflow as tf

flags = tf.app.flags
flags.DEFINE_string("data_dir", None, "Path to the main dir (directory).")
flags.DEFINE_string("output_path", None, "Path to the list txt file.")
FLAGS = flags.FLAGS


def main(_):
    data_dir = FLAGS.data_dir
    output_path = FLAGS.output_path
    class_names_to_ids = {"cat": 0, "dog": 1}
    fd = open(output_path, "w")
    for class_name in class_names_to_ids.keys():
        images_list = os.listdir(data_dir + class_name)
        for image_name in images_list:
            fd.write("{}/{} {}\n".format(class_name, image_name, class_names_to_ids[class_name]))
    fd.close()


if __name__ == "__main__":
    tf.app.run()
