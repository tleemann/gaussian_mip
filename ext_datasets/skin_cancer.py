from typing import Any
from sklearn.model_selection import train_test_split
from tqdm import tqdm
import cv2
import torch
import numpy as np
from PIL import Image
import pandas as pd
from torch.utils.data import Dataset
from glob import glob
from os.path import join, splitext, basename

def preprocess():
    def compute_img_mean_std(image_paths):
        """
            computing the mean and std of three channel on the whole dataset,
            first we should normalize the image from 0-255 to 0-1
        """

        img_h, img_w = 224, 224
        imgs = []
        means, stdevs = [], []

        for i in tqdm(range(len(image_paths)//10)):
            img = cv2.imread(image_paths[i])
            img = cv2.resize(img, (img_h, img_w))
            imgs.append(img)

        imgs = np.stack(imgs, axis=3)
        imgs = imgs.astype(np.float32) / 255.

        for i in range(3):
            pixels = imgs[:, :, i, :].ravel()  # resize to one row
            means.append(np.mean(pixels))
            stdevs.append(np.std(pixels))

        means.reverse()  # BGR --> RGB
        stdevs.reverse()

        return means, stdevs

    def get_duplicates(x):
        unique_list = list(df_undup['lesion_id'])
        if x in unique_list:
            return 'unduplicated'
        else:
            return 'duplicated'
    
    # This set will be df_original excluding all rows that are in the test set
    # This function identifies if an image is part of the train or test set.
    def get_test_rows(x):
        # create a list of all the lesion_id's in the test set
        test_list = list(df_test['image_id'])
        if str(x) in test_list:
            return 'test'
        else:
            return 'train'

    # define the base directory
    base_dir = join('/mnt/ssd3/tobias/mi_auditing/', 'Skin Cancer')
    # compute the mean and std of the images
    all_image_path = glob(join(base_dir, '*', '*.jpg'))
    norm_mean, norm_std = compute_img_mean_std(all_image_path)
    imageid_path_dict = {splitext(basename(x))[0]: x for x in all_image_path}
    lesion_type_dict = {
        'nv': 'Melanocytic nevi',
        'mel': 'dermatofibroma',
        'bkl': 'Benign keratosis-like lesions ',
        'bcc': 'Basal cell carcinoma',
        'akiec': 'Actinic keratoses',
        'vasc': 'Vascular lesions',
        'df': 'Dermatofibroma'
    }
    # read the metadata
    df_original = pd.read_csv(join(base_dir, 'HAM10000_metadata.csv'))
    df_original['path'] = df_original['image_id'].map(imageid_path_dict.get)
    df_original['cell_type'] = df_original['dx'].map(lesion_type_dict.get)
    df_original['cell_type_idx'] = pd.Categorical(df_original['cell_type']).codes
    df_original[['cell_type_idx', 'cell_type']].sort_values('cell_type_idx').drop_duplicates()
    # this will tell us how many images are associated with each lesion_id
    df_undup = df_original.groupby('lesion_id').count()
    # now we filter out lesion_id's that have only one image associated with it
    df_undup = df_undup[df_undup['image_id'] == 1]
    df_undup.reset_index(inplace=True)
    # create a new colum that is a copy of the lesion_id column
    df_original['duplicates'] = df_original['lesion_id']
    # apply the function to this new column
    df_original['duplicates'] = df_original['duplicates'].apply(get_duplicates)
    df_undup = df_original[df_original['duplicates'] == 'unduplicated']
    y = df_undup['cell_type_idx']
    _, df_test = train_test_split(df_undup, test_size=0.2, random_state=101) # stratify=y
    # identify train and val rows
    # create a new colum that is a copy of the image_id column
    df_original['train_or_val'] = df_original['image_id']
    # apply the function to this new column
    df_original['train_or_val'] = df_original['train_or_val'].apply(get_test_rows)
    # filter out train rows
    df_train = df_original[df_original['train_or_val'] == 'train']
    # Copy fewer class to balance the number of num_classes classes
    data_aug_rate = [15,10,5,50,0,40,5]
    #for i in range(num_classes):
    #    if data_aug_rate[i]:
    #        df_train=pd.concat([df_train] + [df_train.loc[df_train['cell_type_idx'] == i,:]]*(data_aug_rate[i]-1), ignore_index=True)
    df_train = df_train.reset_index()
    df_test = df_test.reset_index()
    # return the dataframes
    return (df_train, df_test), (norm_mean, norm_std)

class SkinCancerDataset(Dataset):
    def __init__(self, df, transform=None):
        self.df = df
        self.transform = transform
        
    def __len__(self):
        return len(self.df)
    
    def __getitem__(self, index: int) -> Any:
        X = Image.open(self.df['path'].iloc[index])
        y = torch.tensor(int(self.df['cell_type_idx'].iloc[index]))
        if self.transform:
            X = self.transform(X)
        return X, y
        """img_path = os.path.join(self.img_dir, self.metadata.iloc[index, 1], '.jpg')
        image = read_image(img_path)
        label = self.metadata.iloc[index, 2]
        if self.transform:
            image = self.transform(image)
        if self.target_transform:
            label = self.target_transform(label)
        return image, label"""