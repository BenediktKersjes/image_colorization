import os
import random

from config import images_path

path = images_path + 'validation/test/'
files = os.listdir(path)
shuffled = random.sample(files, k=len(files))

f = open('./data/shuffled.txt', 'x')

for index, file in enumerate(shuffled):
    if file == 'images.txt':
        continue

    f.write(str(index) + ' ' + file + '\n')
