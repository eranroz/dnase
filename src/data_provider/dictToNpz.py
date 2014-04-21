"""
Script for transforming compressed pickle files to npz
compressed pickle file is a compressed (less hard drive needed) but requires more memory and CPU to open
"""
from config import OTHER_DATA
from data_provider import SeqLoader
import argparse

__author__ = 'Eran'

import os
import numpy as np


def pklToNpz(in_dir, out_dir):
    """
    Transforms directory of pkl files to npz files
    @param in_dir: directory to search for pkl files
    @param out_dir: directory to save npz files
    """
    i = 0
    for root, dirs, files in os.walk(in_dir, followlinks=True):
        for f in files:
            if not f.endswith('.pkl'):
                continue
            i += 1
            src_path = os.path.join(root, f)
            res_path = os.path.join(root.replace(in_dir, out_dir), f.replace('.pkl', '.npz'))
            if os.path.exists(res_path):
                continue
            if not os.path.exists(os.path.dirname(res_path)):
                os.makedirs(os.path.dirname(res_path))

            print('%i. %s' % (i, res_path), end="")
            dataDict = SeqLoader.load_result_dict(src_path)
            SeqLoader.save_result_dict(res_path, dataDict)
            print('\tCompressed saved')
    print('Completed')


def check_npz_directory(in_dir):
    try:
        for root, dirs, files in os.walk(in_dir, followlinks=True):
            for f in files:
                if not f.endswith('.npz'):
                    continue
                full_name = os.path.join(root, f)
                data = SeqLoader.load_result_dict(full_name)
                # access to one of the files
                some_in_data = data['chr20'].shape
    except Exception as ex:
        print('Error while opening file: %s. Error: %s' % (full_name, str(ex)))
        raise ex


if __name__ == "__main__":
    parser = argparse.ArgumentParser()

    parser.add_argument('inDir', help="Directory with pkl files to transform to npz")  #, default=MEAN_MARKERS
    parser.add_argument('--outDir', help="Directory to store npz files")  #, default=MEAN_MARKERS_NEW
    parser.add_argument('--check', action="store_true", help="Check directory of npz files (validate they are working")
    args = parser.parse_args()
    if args.outDir is None:
        args.outDir = args.inDir
    if args.check:
        check_npz_directory(args.inDir)
    else:
        pklToNpz(args.inDir, args.outDir)
