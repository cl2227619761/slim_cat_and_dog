# slim_cat_and_dog
## full guide of slim usage

1. generate_list.py
这个文件的作用是将图片的路径和标签转换为txt形式，便于后续生成tfrecord文件使用

2. generate_tfexample.py
这个文件的作用是将图片和标签转换为tfrecord文件

3. tf_to_dataset.py
这个文件的作用是将前面产生的tfrecord文件使用slim读取数据的方式进行读取

4. model.py
这个文件是利用slim搭建的一个简单的CNN分类器

5. train.py
这个文件是用来进行训练的

6. labels.txt
这个文本文件里面是标签对应的名字
