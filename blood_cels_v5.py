import numpy as np  # linear algebra
import pandas as pd  # data processing, CSV file I/O (e.g. pd.read_csv)
import os
#'''
import seaborn as sns
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
import matplotlib.ticker as ticker
import random
import scipy
import shutil

from pprint import pprint
from glob import glob
from pathlib import Path
from keras.preprocessing.image import ImageDataGenerator, array_to_img, img_to_array, load_img
from keras.applications.vgg16 import VGG16 as PTModel, preprocess_input
from sklearn.model_selection import train_test_split
#'''
os.listdir('C:/Users/Андрей/Desktop/комп_зрение/проект_1/task_1_v5/tmp/blood-cells/dataset-master/')
print(os.listdir('C:/Users/Андрей/Desktop/комп_зрение/проект_1/task_1_v5/tmp/blood-cells/dataset-master/'))
PATH = Path('C:/Users/Андрей/Desktop/комп_зрение/проект_1/task_1_v5/tmp/blood-cells/dataset-master/')


# 5
labels_df = pd.read_csv(PATH.joinpath('labels.csv'))
labels_df = labels_df.dropna(subset=['Image', 'Category']) # drop columns that we don't use
labels_df['Image'] = labels_df['Image'].apply(
    lambda x: 'BloodImage_0000' + str(x) + '.jpg'
    if x < 10
    else ('BloodImage_00' + str(x) + '.jpg' if x > 99 else 'BloodImage_000' + str(x) + '.jpg')
)
labels_df = labels_df[['Image', 'Category']]
labels_df.head(15)


# 6
all_image_paths = {
    os.path.basename(x): x for x in glob(
        os.path.join(PATH, '*', '*.jpg')
    )
}
# all_image_paths = [x for x in p.glob('**/*.jpg')]
print('Scans found:', len(all_image_paths), ', Total Headers', labels_df.shape[0])


# 7
labels_df['image_path'] = labels_df['Image'].map(all_image_paths.get)
labels_df.head(10)


# 8
count_of_labels_per_cat = labels_df.Category.value_counts()
to_remove_cat = count_of_labels_per_cat[count_of_labels_per_cat < 10].index
df_next = labels_df.replace(to_remove_cat, np.nan)
df = df_next.dropna()
print('-'*30)
print(df.Category.value_counts())
df.head(12)
print('-'*30)

# 9 train ---------------------------------------------------------------------------------------------------------
train_df, test_df = train_test_split(
    df,
    test_size = 0.30,
    stratify = df['Category']
)
print('shape of data split: ', 'train:', f'{train_df.shape}', 'test:', f'{test_df.shape}')


# 10
print(train_df.Category.value_counts(), '\n')
print(test_df.Category.value_counts())


# попытка_1


