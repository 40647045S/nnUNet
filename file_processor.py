import os
from os import listdir
from os.path import isfile, join
from shutil import copyfile
from tqdm import tqdm
import json
import sys

train_dir = sys.argv[1]
test_dir = sys.argv[2]

os.makedirs('./nnUNet/nnUNet_preprocessed', exist_ok=True)
os.makedirs('./nnUNet/nnUNet_raw', exist_ok=True)
os.makedirs('./nnUNet/nnUNet_trained_models', exist_ok=True)

train_data = []

train_image_dir = join(train_dir, 'image')
train_label_dir = join(train_dir, 'label')

for file in sorted(listdir(train_image_dir)):
    image_path = join(train_image_dir, file)
    label_path = join(train_label_dir, file)
    
    image_name = image_path.split('/')[-1]
    label_name = label_path.split('/')[-1]
    
    assert image_name == label_name
    
    if isfile(image_path) and isfile(label_path):
        train_data.append({'image': image_path, 'label': label_path})

bad_indexes = [189, 262, 659, 662, 703, 744, 1005, 1245, 1347, 1546, 1562, ]

training_files = []

nnu_image_dir = './nnUNet/nnUNet_raw/nnUNet_raw_data/Task101_BrainTS/imagesTr/'
nnu_label_dir = './nnUNet/nnUNet_raw/nnUNet_raw_data/Task101_BrainTS/labelsTr/'

os.makedirs(nnu_image_dir, exist_ok=True)
os.makedirs(nnu_label_dir, exist_ok=True)

for index, sample in enumerate(tqdm(train_data), start=1):
    
    if index in bad_indexes:
        continue
    
    old_filename = sample['image'].split('/')[-1]
    
    image_filename = f'BrainTS_{index:04d}_0000.nii.gz'
    label_filename = f'BrainTS_{index:04d}.nii.gz'
    
    copyfile(sample['image'], join(nnu_image_dir, image_filename))
    copyfile(sample['label'], join(nnu_label_dir, label_filename))
    
    training_files.append({'image': join(nnu_image_dir, label_filename),
                           'label': join(nnu_label_dir, label_filename)})

test_data = []

for file in sorted(listdir(test_dir)):
    test_path = join(test_dir, file)
    
    test_name = test_path.split('/')[-1]
    
    if isfile(test_path):
        test_data.append(test_path)

test_files = []
uun_test_dir = './test_images'
os.makedirs(uun_test_dir, exist_ok=True)

for index, test_path in enumerate(tqdm(test_data)):
    old_filename = test_path.split('/')[-1]
    new_filename = old_filename.split('.')[0]+'_0000.nii.gz'
    
    copyfile(test_path, join(uun_test_dir, new_filename))

dataset_config = dict(
    name='BrainTS',
    description='abc',
    tensorImageSize='4D',
    modality={'0':'MRI'},
    labels={'0':'background', '1':'tumor'},
    numTraining=len(training_files),
    numTest=0,
    training=training_files,
    test=[],
)

with open('./nnUNet/nnUNet_raw/nnUNet_raw_data/Task101_BrainTS/dataset.json', 'w') as file:
    json.dump(dataset_config, file)
