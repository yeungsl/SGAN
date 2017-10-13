import os, sys, re, urllib.request, tarfile
import os.path
import numpy as np
import matplotlib.pyplot as plt
from inspect import getsourcefile

current_path = os.path.abspath(getsourcefile(lambda:0))
parent_dir = os.path.split(os.path.dirname(current_path))[0]
sys.path.insert(0, parent_dir)



def download_and_extract(dest_dir):
    DATA_URL = "https://www.cs.toronto.edu/~kriz/cifar-10-python.tar.gz"

    # download the cifar-10 data set

    if not os.path.exists(dest_dir):
        os.makedirs(dest_dir)
    filename = DATA_URL.split('/')[-1]
    filepath = os.path.join(dest_dir, filename)
    if not os.path.exists(filepath):
        def _progress(count, block_size, total_size):
            sys.stdout.write('\r >> Downloading %s %.1f%%' %(filename, float(count*block_size)/float(total_size)*100.0))
            sys.stdout.flush()
        filepath, _ = urllib.request.urlretrieve(DATA_URL, filepath, _progress)

    print()
    statinfo = os.stat(filepath)
    print('Successfully downloaded', filename, statinfo.st_size, 'bytes.')

    with tarfile.open(filepath, 'r:gz') as t:
        dataset_dir = os.path.join(dest_dir, t.getmembers()[0].name)
        t.extractall(dest_dir)

    return dataset_dir

data_dir = download_and_extract('./cifar-10_data')
print(data_dir)

