"""
Script for transforming compressed pickle files to npz
compressed pickle file is a compressed (less hard drive needed) but requires more memory and CPU to open
"""

import argparse
from data_provider import SeqLoader

__author__ = 'Eran'

import os
import zlib


def pklToNpz(in_dir, out_dir):
    """
    Transforms directory of pkl files to npz files
    @param in_dir: directory to search for pkl files
    @param out_dir: directory to save npz files
    """
    i = 0
    try:
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
                try:
                    data_dict = SeqLoader.load_result_dict(src_path)
                    SeqLoader.save_result_dict(res_path, data_dict)
                except zlib.error:
                    print('\tfailed- corrupted file')
                    continue
                print('\tCompressed saved')
    except KeyboardInterrupt:
        # delete the last file - to prevent partial corrupted data
        if os.path.exists(res_path):
            os.remove(res_path)
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

    parser.add_argument('inDir', help="Directory with pkl files to transform to npz")
    parser.add_argument('--outDir', help="Directory to store npz files")
    parser.add_argument('--check', action="store_true", help="Check directory of npz files (validate they are working")
    args = parser.parse_args()
    if args.outDir is None:
        args.outDir = args.inDir
    if args.check:
        check_npz_directory(args.inDir)
    else:
        pklToNpz(args.inDir, args.outDir)
