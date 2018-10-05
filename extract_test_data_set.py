import os
import random
import glob

from config import images_path


# get out of folder
all_files = [os.path.relpath(os.path.join(root, f), '.')
             for root, dirs, files in os.walk(images_path + 'train/') for f in files]

for one_file in all_files:
    one_file_new = one_file.replace('\\0000', '0000').replace('/0000', '0000')
    os.rename(one_file, one_file_new)

# sample test dataset
all_files = glob.glob(images_path + 'train/*') 
test_files = random.sample(all_files, int(len(all_files) / 10))

os.mkdir(images_path + 'test/')
for test_file in test_files:
    _, file_name = os.path.split(test_file)
    os.rename(test_file, images_path + 'test/' + file_name)
