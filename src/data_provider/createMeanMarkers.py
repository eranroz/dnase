"""
Script for transforming directory of wig files of specific cell type
to a pkl file with mean of all samples within directory
"""
import argparse
import os
import subprocess

import numpy as np

from config import WIG_TO_BIG_WIG, BIG_WIG_TO_BED_GRAPH, CHROM_SIZES
from data_provider import SeqLoader

#from multiprocessing.pool import Pool
#_pool_process = Pool(2)

__author__ = 'eranroz'


def transform_directory(input_dir, output_dir, skip_working=False):
    """

    @param input_dir: Directory to transform
    @param output_dir: Destination directory for transformations
    @param skip_working: skip on directories with temp .bg files (ot prevent clashing of different processes)
    """
    import random

    #global _pool_process
    files = [f for f in os.listdir(input_dir)]
    random.shuffle(files)  # shuffle so it won't get stacked with other processes
    if len(files) == 0:
        return
    print('trasnforming dir: %s=>%s ' % (input_dir, output_dir))
    first_file = os.path.join(input_dir, files[0])
    if os.path.isdir(first_file):
        # transform subdirectories
        for dir in files:
            dir_in = os.path.join(input_dir, dir)
            dir_out = os.path.join(output_dir, dir)
            if not os.path.exists(dir_out):
                print('creating dir %s' % dir_out)
                os.mkdir(dir_out)
            transform_directory(dir_in, dir_out)
    else:
        # change wig to bg
        mean_name = os.path.join(output_dir, 'mean.npz')
        if os.path.exists(mean_name):
            print('Skipping %s\t already have mean file' % output_dir)
            return
        is_other_working = any([True for f in files if f.endswith('.bg')])
        if is_other_working and skip_working:
            return
        print('trasnforming files: %s=>%s ' % (input_dir, output_dir))
        #pool_process = Pool(2)
        #_pool_process.map(do_transform, getTransforms(input_dir, output_dir))
        for a in get_transforms(input_dir, output_dir):
            wig_to_bed_graph(a)
        # join bgs and delete them
        #samples = []
        chrom_samples = dict()
        n_samples = 0.0
        for f in os.listdir(output_dir):
            if not f.endswith('.bg'):
                continue
            print(f)
            n_samples += 1.0
            sample = SeqLoader.load_bg(os.path.join(output_dir, f))
            #samples.append(sample)
            for k, v in sample.items():
                if k not in chrom_samples:
                    chrom_samples[k] = v
                else:
                    chrom_size = max(len(chrom_samples[k]), len(v))
                    new_mean = np.zeros(chrom_size)
                    new_mean[0:len(chrom_samples[k])] = chrom_samples[k]
                    new_mean[0:len(v)] += v
                    chrom_samples[k] = new_mean
        print('Finished loading samples')
        mean_marker = dict()
        for chrom in chrom_samples.keys():
            #max_len = max([len(s[chrom]) for s in samples])
            #join_mat = np.zeros((len(samples), max_len))
            #for samp_i, samp in enumerate(samples):
            #    join_mat[samp_i, 0:len(samp[chrom])] = samp[chrom]
            mean_marker[chrom] = chrom_samples[chrom] / n_samples

        print('Saving mean to: %s' % mean_name)
        SeqLoader.save_result_dict(mean_name, mean_marker)
        print('saved deleting files in %s' % output_dir)
        to_del = list(os.listdir(output_dir))
        for f in to_del:
            if not f.endswith('.npz'):
                os.remove(os.path.join(output_dir, f))


def wig_to_bed_graph(in_out):
    """
    @param in_out: tuple of input output paths
    """
    in_file, out_file = in_out
    if os.path.exists(out_file):  # skip if already exist
        return

    bw_file = out_file.replace('.bg', '.bw')
    print('%s =>%s ' % (in_file, bw_file))
    subprocess.call([WIG_TO_BIG_WIG, in_file, CHROM_SIZES, bw_file])
    print('%s =>%s ' % (bw_file, out_file))
    subprocess.call([BIG_WIG_TO_BED_GRAPH, bw_file, out_file])
    try:
        os.remove(bw_file)
    except:
        pass  # just ignore ;)


def get_transforms(input_dir, output_dir):
    """
    Scans directory and sub directories and returns tuples of input and output
    @param input_dir:
    @param output_dir:
    """
    for s_name in os.listdir(input_dir):
        i_name = os.path.join(input_dir, s_name)
        o_name = os.path.join(output_dir, s_name.replace('.gz', '').replace('.wig', '.bg'))
        if os.path.isdir(i_name):
            for transform in get_transforms(i_name, o_name):
                yield transform
        else:
            if not os.path.exists(o_name):
                yield (i_name, o_name)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('input', help="Input directory")
    parser.add_argument('output', help="Output directory")
    parser.add_argument('--skip', help="Skip directories that have bg files", default=False, type=bool)
    args = parser.parse_args()
    print((args.skip, args.input, args.output))
    transform_directory(args.input, args.output, args.skip)
