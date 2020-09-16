import os
import numpy as np
import sys
import time
import nsml 
from nsml.constants import DATASET_PATH


def test_path_loader (root_path): ## 
    image_path = []
    image_keys = []
    for _,_,files in os.walk(os.path.join(root_path,'test_data')):
        for f in files:
            path = os.path.join(root_path,'test_data',f)
            if path.endswith('.png'):
                image_keys.append(int(f[:-4]))
                image_path.append(path)

    return np.array(image_keys), np.array(image_path)


def feed_infer(output_file, infer_func):
    root_path = os.path.join(DATASET_PATH,'test')
    image_keys, image_path = test_path_loader(root_path)
    result = infer_func(image_path)
    print('write output')

    with open(output_file, 'wt') as file_writer:
        for key, value in zip(image_keys,result):
            file_writer.write('{} {}\n'.format(key, value))
    if os.stat(output_file).st_size ==0:
        raise AssertionError('output result of inference is nothing')