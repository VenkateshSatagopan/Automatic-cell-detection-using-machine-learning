# -*- coding: utf-8 -*-
"""
Created on Thu Jan  2 17:01:34 2020

@author: venkatesh
"""

import tensorflow as tf
import numpy as np
from PIL import Image
import os
import shutil
import argparse
import glob



parser = argparse.ArgumentParser()
parser.add_argument('--input-folder', type=str, default='input/', help='source')  # input file/folder, 0 for webcam
parser.add_argument('--output-folder', type=str, default='patches', help='output folder')  # output folder
parser.add_argument('--patch-size', type=int, default=416, help='inference size (pixels)')
parser.add_argument('--num-patches', type=int, default=30, help='Number of patches to be extracted')
opt = parser.parse_args()
print(opt)
if os.path.exists( opt.output_folder ):
    shutil.rmtree( opt.output_folder )  # delete output folder
os.makedirs( opt.output_folder )


filenames = glob.glob('%s/*.*' % opt.input_folder) # filenames containing the big images
if not filenames:
   num_samples=0
   print("Choose the input path correctly")
   print("No of input files is present is zero")
else:
   print(filenames)
   num_samples = len(filenames)

#print(num_samples)
num_parallel_calls=4

def save_image(path, patch):
 for val in range(0,patch.shape[0]):
    im = Image.fromarray(patch[val,:,:,:])
    im.save(path+'patch_'+str(val+1)+'.png')


def get_patches(image, num_patches=100, patch_size=16):
    """Get `num_patches` random crops from the image"""
    patches = []
    for i in range(num_patches):
        patch = tf.random_crop(image, [patch_size, patch_size, 3])
        patches.append(patch)

    patches = tf.stack(patches)
    assert patches.get_shape().dims == [num_patches, patch_size, patch_size, 3]

    return patches

def read_image_fn(filename):
    """Read the image"""
    image_string = tf.read_file(filename)

    # Decode the image
    image_decoded = tf.image.decode_png(image_string, channels=3)

    return image_decoded



# Procedure for creating patches using tf.dataset API

get_patches_fn = lambda image: get_patches(image, num_patches=opt.num_patches, patch_size=opt.patch_size)


# Create a Dataset serving batches of random patches in our images
if num_samples:
  dataset = (tf.data.Dataset.from_tensor_slices(filenames) # Step 1: create Tensorflow slices
            .map(read_image_fn,num_parallel_calls=num_parallel_calls)# Step 2: Read images
            .map(get_patches_fn, num_parallel_calls=num_parallel_calls)  # Step 3: Generate patches
            .prefetch(1)  # Step 4: make sure you always have one batch ready to serve
            )

  iterator = dataset.make_one_shot_iterator()
  patches = iterator.get_next()


if __name__ == '__main__':
    init = tf.global_variables_initializer()
    with tf.Session() as sess:
        
        for i in range(0,num_samples):
            output_patches = sess.run( patches )
            output_patches = np.asarray( output_patches )

            print(output_patches.shape)
            save_image(opt.output_folder + '/' + str( i + 1 )+'_', output_patches )

