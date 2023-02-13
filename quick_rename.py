# I accidentally included a redundant ".jpg" in the middle of hundreds of thousands of filenames.
# This is the code to fix it -.-

import os

path = 'dummy_dataset/'

#print(os.listdir(path))

test_fn = "20230109_175119.jpg_0_4.jpg"

def cut_down(fn):
    fn = fn.split('.')
    middle = fn[1][3:]
    return fn[0] + middle + '.' + fn[2]

dirs = [path+i+'/' for i in os.listdir(path) if os.path.isdir(path+i)]
for d in dirs:
    for image in os.listdir(d):
        print(image)
        os.rename(d+image, d+cut_down(image))
