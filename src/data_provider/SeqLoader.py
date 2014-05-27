"""
General utility class for accessing pickle data files and for simple transformations

Generally data is stored as dict of arrays, as dumped and compressed file.
using pytables is probably the best approach but we may have problems with portability
(or easy installation in different environments)
so here we use simpler approach with more standard packages zlib and pickle
"""
import gzip
from io import BufferedReader
import io
import os
import pickle
import zlib
import logging

import numpy as np

from config import DATA_DIR, PUBLISH_DIR, MEAN_MARKERS, PUBLISH_URL_PATH, BED_GRAPH_TO_BIG_WIG, CHROM_SIZES
import re

__author__ = 'eran'


def wig_transform(wig_file, smoothing=100, output=True):
    """
    Gets a wig file and transforms it to dictionary (chr1 -> [scores], ch2 -> [scores]...)
    and saves it with pickle
    @param output: Whether to output the transformed wig to npz file (name.smoothing.npz)
    @param smoothing: bin size for smoothing (common raw data can be 20bp)
    @param wig_file: name of wig file (or wig.gz)
    @return: a dictionary where key are chromosomes and values are scores

    @note it is recommended to use use bed graph files and use load_bg which works much faster
    """
    written_dict = dict()  # dictionary keys: chromosomes [ score, score], and chromosome-position: start position

    if wig_file.endswith('.gz'):
        in_stream = BufferedReader(gzip.open(wig_file, 'rb'))
        in_file = (in_line.decode('ascii') for in_line in in_stream)
    else:
        in_stream = io.open(wig_file, 'r')

    prev_data = []
    l_prev_data = 0
    chrm_score = []
    chrom = None
    rlines = 0
    frag_size = 20
    cur_smoothing = smoothing / frag_size
    prev_pos = 0
    for r in in_file:
        if r[0] == 'v':  # new chromosome, variable step
            if chrom is not None:
                chrm_score.append(sum(prev_data))
                written_dict[chrom] = chrm_score

            chrm_score = []
            prev_data = []
            print(r.strip())
            chrom_track = r.rstrip().split(' ')
            frag_size = int(chrom_track[2].split('=')[1].strip())
            cur_smoothing = int(smoothing / frag_size)
            chrom = chrom_track[1].split('=')[1]
            prev_pos = 1
            pos, score = next(in_file).split('\t')
        elif r[0] == 't':
            continue  # track line
        else:
            pos, score = r.split('\t')
        pos = int(pos)
        score = int(score)
        if pos != prev_pos:
            prev_data += [0] * int(np.floor((pos - prev_pos) / frag_size))
            l_prev_data = len(prev_data)
        prev_data += [score]
        l_prev_data += 1
        prev_pos = pos + frag_size
        if l_prev_data > cur_smoothing:
            for smooth_bin in zip(*[iter(prev_data)] * cur_smoothing):
                chrm_score.append(sum(smooth_bin))
            remaining = (len(prev_data) % cur_smoothing)
            prev_data = prev_data[-remaining:] if remaining > 0 else []
            l_prev_data = len(prev_data)
        rlines += 1

    if len(prev_data) != 0:
        chrm_score.append(sum(prev_data))
    written_dict[chrom] = chrm_score

    in_stream.close()
    print('Finished reading and down-sampling')

    if output:
        # save to same file but with different extensions
        wig_file = wig_file.replace('.gz', '')  # remove .gz extension for compressed file
        output_filename = wig_file.replace('.wig', '.%i.npz' % smoothing)
        save_result_dict(output_filename, output_filename)
        #with open(output_filename, 'wb') as output:
        #    output.write(zlib.compress(pickle.dumps(written_dict, pickle.HIGHEST_PROTOCOL)))
        print('Finished writing file')

    return written_dict


def bed_graph_transform(bedgraph_file, smoothing=100, save=True):
    """
    Gets a bedgraph file and transforms it to dictionary (chr1 -> [scores], ch2 -> [scores]...)
    and saves it with pickle
    @param save:
    @param smoothing:  bin size for smoothing (common raw data can be 20bp)
    @param bedgraph_file: input file
    @return:
    """
    # TODO: may be duplicant to load_bg below

    written_dict = dict()  # dictionary keys: chromosomes [ score, score], and chromosome-position: start position
    chrm_score = []  # in size of smoothing
    max_resolution = 20  # the minimum size of resolution given
    with open(bedgraph_file, 'r') as bgfile:
        prev_data = []  # in max resolution size
        prev_chrom, prev_start, prev_end, score = bgfile.readline().rstrip().split('\t')
        prev_start = int(prev_start)
        prev_end = int(prev_end)
        score = float(score)
        # add zeros to beginning until start
        for pos in range(1, prev_end, smoothing):
            if pos < prev_start:
                chrm_score.append(0)
            elif pos < prev_start + smoothing:  # begin - less than smoothing size
                chrm_score.append(score * (pos - prev_start))
            elif prev_end - pos < smoothing:  # end - less than smoothing size
                prev_data = [max_resolution * score] * int((prev_end + 1 - pos) / max_resolution)
            else:
                chrm_score.append(score * smoothing)

        written_dict[prev_chrom + '-position'] = prev_start
        rlines = 0
        #longest = 1000
        smoothing /= max_resolution
        if int(smoothing) != smoothing:
            raise Exception("bad smoothing parameter. (should be divided by 20)")
        else:
            smoothing = int(smoothing)
        for r in bgfile:
            chrom, start, end, score = r.rstrip().split('\t')
            if chrom != prev_chrom:
                if len(prev_data) != 0:
                    chrm_score.append(sum(prev_data))
                written_dict[prev_chrom] = chrm_score
                chrm_score = []
                written_dict[chrom + '-position'] = int(start)
                print(r)
                prev_data = []
                prev_end = 0
            start = int(start)
            end = int(end)
            score = float(score)
            if start != prev_end:
                prev_data += [0] * int((start - prev_end) / max_resolution)

            if rlines % 100000 == 0:
                print(rlines)

            prev_data += [max_resolution * score] * (int((end - start) / max_resolution))
            prev_end = end
            prev_chrom = chrom
            for smooth_bin in zip(*[iter(prev_data)] * smoothing):
                chrm_score.append(sum(smooth_bin))
            remaining = (len(prev_data) % smoothing)
            prev_data = prev_data[-remaining:] if remaining > 0 else []
            rlines += 1
        if len(prev_data) != 0:
            chrm_score.append(sum(prev_data))
        written_dict[prev_chrom] = chrm_score

    print('Finished reading and down-sampling')
    if save:  # save to file or return as is
        output_filename = bedgraph_file.replace('.bedGraph', '.' + str(smoothing) + '.npz')
        save_result_dict(output_filename, written_dict)
        print('Finished writing file')
    else:
        return written_dict


def chrom_sizes():
    """
    Get chromosome sizes of hg19 as dictionary
    @rtype : dict
    @return: dictionary with key as chromosome name and value as size (for bins of size of 20)
    """
    with open(CHROM_SIZES) as chrom_size_fd:
        chromosomes = (r.split('\t') for r in chrom_size_fd.readlines())
    chrom_dict = dict(((chrom[0], int(chrom[1]) // 20) for chrom in chromosomes))
    return chrom_dict


def load_bg(bg_file):
    """
    Loads bed graph
    should be faster then the above and directly load it
    @param bg_file: bg file to be loaded
    @return:  dictionary of chromosomes keys and values as wig like arrays
    """

    def load_bg_slow():
        """
        Fallback - if pandas doesnt exist and you get into trouble with cython compilations
        """
        chr_to_ind = dict()
        chr_to_ind_inv = dict()
        for i in range(1, 26):
            chr_name = 'chr%i' % i
            if i == 23:
                chr_name = 'chrX'
            if i == 24:
                chr_name = 'chrY'
            if i == 25:
                chr_name = 'chrM'
            chr_name = chr_name.encode('ascii')
            chr_to_ind[chr_name] = i
        for k, v in chr_to_ind.items():
            chr_to_ind_inv[v] = k.decode('utf-8')
        # 0 - chromosome, 1 - start, 2 - end
        bed_matrix = np.loadtxt(bg_file, delimiter='\t', usecols=(0, 1, 2, 3),
                                converters={
                                    0: lambda x: chr_to_ind[x]
                                }, dtype=int)
        chromosomes_ind = set(bed_matrix[:, 0])
        res_dict = dict()
        for chromosome in chromosomes_ind:
            selector = bed_matrix[:, 0] == chromosome
            chrom_matrix = np.array(bed_matrix[selector, [1, 2, 3]])
            # TO BIN SIZE
            chrom_matrix[:, [0, 1]] = chrom_matrix[:, [0, 1]] / 20
            last_end = max(chrom_matrix[:, 1])
            long_rep = np.zeros(last_end)
            for line in chrom_matrix:
                long_rep[line[0]:line[1]] = line[2]
            res_dict[chr_to_ind_inv[chromosome]] = long_rep
        return res_dict

    def load_bg_pandas():
        """
        Load bed graph using pandas
        """
        import pandas as pd

        data = pd.read_csv(bg_file, sep='\t', header=None)
        group_by_chr = data.groupby(0)
        res_dict = dict()
        for k, v in group_by_chr:
            chrom_matrix = v[[1, 2, 3]].values
            # TO BIN SIZE
            chrom_matrix[:, [0, 1]] = chrom_matrix[:, [0, 1]] / 20
            last_end = max(chrom_matrix[:, 1])
            long_rep = np.zeros(last_end)
            rows_sel = chrom_matrix[:, 0] > 0
            sig_len = 0
            indics_arr = np.array(chrom_matrix, dtype=int)
            while np.any(rows_sel):
                indics = indics_arr[:, 0] + sig_len
                long_rep[indics] = indics_arr[:, 2]
                sig_len += 1
                rows_sel = (indics_arr[:, 1] - indics_arr[:, 0]) > sig_len
                indics_arr = indics_arr[rows_sel, :]

            res_dict[k] = long_rep
        return res_dict

    try:
        from pyx import BedGraphReader

        return BedGraphReader.load_bedgraph(bg_file, chrom_sizes())
    except ImportError:
        logging.warn("couldn't load BedGraphReader. reading bg files could be slow. Compile pyx directory/get pandas")
        try:
            return load_bg_pandas(bg_file)
        except ImportError:
            return load_bg_slow(bg_file)


def save_result_dict_old(output_filename, written_dict):
    """
    Saves a result dictionary to file for later use
    @param output_filename: filename for the saved dictionary
    @param written_dict: dictionary to save
    @return:
    """
    with open(output_filename, 'wb') as output:
        output.write(zlib.compress(pickle.dumps(written_dict, pickle.HIGHEST_PROTOCOL)))


def save_result_dict(output_filename, written_dict):
    """
    Saves a result dictionary to file for later use
    @param output_filename: filename for the saved dictionary
    @param written_dict: dictionary to save
    @return:
    """
    np.savez_compressed(output_filename, **written_dict)


def load_result_dict(output_filename):
    """
    Saves a result dictionary to file for later use
    @param output_filename: filename for the saved dictionary
    @return: result dictionary
    """
    if output_filename.endswith('.npz'):
        return np.load(output_filename)
    else:
        with open(output_filename, 'rb') as file:
            data = zlib.decompress(file.read())
            sequence_dict = pickle.loads(data)
            return sequence_dict


def available_experiments(cell_type, experiments=None):
        """
        @param experiments: name of experiments to check for availability
        @param cell_type: cell type to locate experiments for
        @return: mapping between available experiments and the file path
        @rtype: dict
        @raise Exception:
        """
        warn = experiments is not None
        experiments_to_load = os.listdir(MEAN_MARKERS) if experiments is None else experiments
        available_ex = dict()
        for ex in experiments_to_load:
            ex_dir = os.path.join(MEAN_MARKERS, ex)
            if not os.path.exists(ex_dir):
                raise Exception("Data for experiment %s isn\'t available" % ex)
            ex_cell_type_dir = os.path.join(ex_dir, cell_type)
            # fallback to find similar cell type
            if not os.path.exists(ex_cell_type_dir):
                # may be different naming fetal_brain => brain_fetal
                ex_cell_type_dir = os.path.join(ex_dir, re.sub('fetal_(.+)', '\\1_fetal', cell_type))
                if not os.path.exists(ex_cell_type_dir):
                    if warn:
                        logging.warning('Cell type %s not found for expirment %s' % (cell_type, ex))
                    continue
            mean_file = os.path.join(ex_cell_type_dir, 'mean.npz')
            available_ex[ex] = mean_file
        return available_ex


def load_experiments(cell_type, experiments=None, chromosomes=None, resolution=20):
    """
    Generic function to access experiments data of "markers", such as methylations and acetylations.
    Since it is common that we don't have the experiment data for the specific sample
    (e.g. same tissue, same person) we workaround it with mean scores of various experiments on same tissue
    but different people.
    {MEAN_MARKERS} directory can be populated by data_provider/createMeanMarkers script


    @param cell_type: cell type
    @param experiments: name of experiments
    @param chromosomes:
    @return: Dictionary with key as chromosomes and values as matrix, where rows are different experiments
             and values are columns
    @raise Exception: In case of data not available in MEAN_MARKERS
    @rtype : dict
    """
    from scipy.sparse import lil_matrix, vstack, csr_matrix

    res_dict = dict()
    experiment_mapping = available_experiments(cell_type, experiments)
    existing_experiments = experiment_mapping.keys()
    n_expiremnts = len(list(experiment_mapping.keys()))
    for ex_i, mean_file in enumerate(experiment_mapping.values()):
        mean_data = load_result_dict(mean_file)

        chromosomes_to_load = mean_data.keys() if chromosomes is None else chromosomes
        for chromosome_key in chromosomes_to_load:
            chromosome_value = mean_data[chromosome_key]
            if resolution != 20:
                chromosome_value = down_sample(chromosome_value, resolution//20)
            # add to combined dictionary
            if chromosome_key not in res_dict:
                #combined = np.zeros((n_expiremnts, chromosome_value.shape[0]))
                #combined[ex_i, :] = chromosome_value
                combined = csr_matrix(chromosome_value)
            else:
                existing_val = res_dict[chromosome_key]
                # extend number of columns if necessary
                if existing_val.shape[1] < chromosome_value.shape[0]:
                    #combined = np.zeros((n_expiremnts, chromosome_value.shape[0]))
                    existing_val = existing_val.todense()
                    combined = np.zeros((existing_val.shape[0], chromosome_value.shape[0]))
                    combined[:, 0:existing_val.shape[1]] = existing_val
                    combined = csr_matrix(combined)
                    #combined[:ex_i, 0:existing_val.shape[1]] = existing_val[:ex_i, :]
                else:
                    combined = existing_val

                combined = vstack([combined, csr_matrix(chromosome_value)])
                #combined[ex_i, 0:chromosome_value.shape[0]] = chromosome_value

            res_dict[chromosome_key] = combined

    # for now don't spread sparse matrix around out of the function scope
    for k in res_dict.keys():
        res_dict[k] = res_dict[k].todense()
    return res_dict, existing_experiments


def load_dict(name, resolution=20, transform=None, directory=DATA_DIR, chromosome=None, orig_resolution=20):
    """
    Loads transformed dictionary from DATA_DIR.
    @param orig_resolution: original resolution
    @param directory: directory to load the data from
    @param transform: A transform function/callable object after reading the sequence
    @param resolution: resolution for genome default is 20. can be any number that can be divided by 20
    @param name: name fo file in DATA_DIR
    @param chromosome: chromosomes to load (other chromosomes are ignored)
    @return: dictionary of chromosomes scores and start positions
    """
    if os.path.exists(os.path.join(directory, '%s.%i.npz' % (name, orig_resolution))):
        sequence_dict = load_result_dict(os.path.join(directory, '%s.%i.npz' % (name, orig_resolution)))
    elif os.path.exists(os.path.join(directory, name)):
        sequence_dict = load_result_dict(os.path.join(directory, name))
        logging.warning('loading file %s with no resolution in its name. Assuming 20' % name)
    else:
        file_name = os.path.join(directory, '%s.%i.pkl' % (name, orig_resolution))
        try:
            with open(file_name, 'rb') as file:
                decompress = zlib.decompress(file.read())
                sequence_dict = pickle.loads(decompress)
            # select relevant chromosomes
            dict_keys = [key for key in sequence_dict.keys() if
                         '-position' not in key and (not isinstance(chromosome, list) or key in chromosome)
                         and (not isinstance(chromosome, str) or chromosome == key)]
            # remove the rest
            orig_keys = list(sequence_dict.keys())
            for k in orig_keys:
                if k not in dict_keys:
                    del sequence_dict[k]
        except IOError as err:
            print('Please check: ', os.path.abspath(file_name))
            raise err
    dict_keys = [k for k in sequence_dict.keys() if chromosome is None or k in chromosome]
    res_dict = dict()
    for key in dict_keys:
        seq = sequence_dict[key]
        logging.debug('Preparing %s' % key)
        if resolution != orig_resolution:  # Down-sample sequence
            seq = down_sample(seq, resolution / orig_resolution)  # TODO: would be nice to create lazy evaluated
        if transform is not None:
            seq = transform(seq)
        res_dict[key] = seq

    return res_dict



def continuous_transform(seq):
    """
    Transform the sequence to ln(1+x)
    @param seq: sequence to transform
    """
    return np.log(1 + np.array(seq))


def down_sample(seq, new_sampling_factor=1):
    """
    down samples a sequence with defined factor.
    @param new_sampling_factor: number of bins to combine together
    @param seq: sequence to down sample
    """
    if new_sampling_factor != int(new_sampling_factor):
        raise Exception("Bad sampling factor - validate your resolution")

    seq = np.array(seq)
    # either cut the end (here) or extend with zero (but may later damage the max chromosome length)
    seq = seq[0:seq.shape[0] - (seq.shape[0] % new_sampling_factor)]
    return seq.reshape(seq.shape[0] // new_sampling_factor, new_sampling_factor).sum(1)


def build_bedgraph(classified_seq, resolution, output_file):
    """
    Creates bed graph file from dictionary
    @param classified_seq: Dictionary to transform
    @param resolution: bin resolution
    @param output_file: file or path to file
    """
    if isinstance(output_file, str):
        output = open(output_file, 'w')
    else:
        output = output_file
    for chromosome, chromosome_seq in classified_seq.items():
        prev_val = None
        start = str(1)
        chromosome_seq = np.around(chromosome_seq, decimals=3)
        for i, val in enumerate(chromosome_seq):
            if prev_val is None:
                prev_val = val
            if val != prev_val or i - 1 == len(chromosome_seq):
                prev_val = '%.3f' % prev_val
                pos = str((i + 1) * resolution)
                to_write = '\t'.join([chromosome, start, pos, prev_val]) + '\n'
                output.write(to_write)
                start = pos
            prev_val = val
    if isinstance(output_file, str):
        output.close()


def bg_to_bigwig(bed_graph, big_wig=None):
    """
    Transforms bed graph to big wig
    @param bed_graph: file to transform
    @param big_wig: full output file or default to PUBLISH_DIR
    """
    if big_wig is None:
        big_wig = os.path.split(bed_graph)[-1].replace('.bg', '.bw')
        big_wig = os.path.join(PUBLISH_DIR, big_wig)
        print('Saving to ', big_wig)
    import subprocess

    subprocess.call([BED_GRAPH_TO_BIG_WIG, bed_graph, CHROM_SIZES, big_wig])
    print('Created bw')