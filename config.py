# default config
images_path_base = './data/images/'
dataset = 'MITPlaces365-Standard'
logs_path = 'D:\\dateien\\logs\\'
trained_models_path = 'D:\\dateien\\trained_models\\'
grid_size = 16

# user defined config
try:
    from config_local import *
except ImportError:
    pass

images_path = images_path_base + dataset + '/'
