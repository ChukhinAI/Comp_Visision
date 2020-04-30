import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import os
from os.path import join
import random
import pathlib
import tensorflow as tf
#tf.enable_eager_execution()
#tf.executing_eagerly()
import IPython.display as display

from sklearn.model_selection import train_test_split
from keras_applications import resnet


#tf.random.set_random_seed(42)
np.random.seed(42)

basedir = 'spoon-vs-fork\spoon-vs-fork'
fork_dir = join(basedir, 'fork')
spoon_dir = join(basedir, 'spoon')
spoon_paths = [join(spoon_dir, img_path) for img_path in os.listdir(spoon_dir)]
fork_paths = [join(fork_dir, img_path) for img_path in os.listdir(fork_dir)]
img_paths = spoon_paths + fork_paths
print(len(img_paths))


def load_data(basedir):
    folders = os.listdir(basedir)
    print(folders)
    result = pd.DataFrame(columns=['filename', 'class'])
    for folder in folders:
        files = [join(basedir, folder, file) for file in os.listdir(join(basedir, folder))]
        df = pd.DataFrame({'filename': files, 'class': folder})
        result = pd.concat([result, df])
    return result

image_df = load_data(basedir)


def validate_data(image_df):
    result = image_df.copy()
    allowed_extensions = ['jpg', 'jpeg', 'png', 'gif']
    for img in image_df.filename:
        extension = str.lower(os.path.splitext(img)[1])[1:]
        if extension not in allowed_extensions:
                    result = result[result.filename != img]
                    print("Removed file with extension '{}'".format(extension))
    return result

image_df = validate_data(image_df)



X_train, X_test, y_train, y_test = train_test_split(image_df.filename, image_df['class'], test_size=0.2, random_state=42)


# 8 training


resnet_weights_path = '../input/resnet50/resnet50_weights_tf_dim_ordering_tf_kernels_notop.h5'
resnet = resnet(include_top=False, pooling='avg', weights=resnet_weights_path)

