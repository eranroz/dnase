"""'
Script for download and installation of data and required programs

Some functions requires rsync


@see {transformWig} - another script for transformations
''"""
import argparse
import ftplib
from multiprocessing import Pool
import os
import urllib
import time

from config import DATA_DIR, BIN_DIR, OTHER_DATA, SIGNAL_DIR, WIG_TO_BIG_WIG, BIG_WIG_TO_BED_GRAPH, CHROM_SIZES,\
    RAW_DATA_DIR
from data_provider import SeqLoader


SMOOTHING = 20
BED_GRAPH_DIR = os.path.join(DATA_DIR, 'bedGraph')


def setup_environment():
    """
    Downloads some required programs from UCSC.
    """
    tools = ["fetchChromSizes", "wigToBigWig", "bigWigToBedGraph", "bedGraphToBigWig"]
    try:
        import urllib.request

        urlret = urllib.request.urlretrieve
    except ImportError:
        import urllib.urlretrieve

        urlret = urllib.urlretrieve
    for tool in tools:
        if not os.path.exists(os.path.join(BIN_DIR, tool)):
            urlret("http://hgdownload.cse.ucsc.edu/admin/exe/linux.x86_64/%s" % tool,
                   os.path.join(BIN_DIR, tool))


def download_dnase_human_data(ignore_files=None):
    """
    Connects to genboree Epigenome atlas to download specific cell types
    chromatin accessibility experiment results
    @param ignore_files:  files to ignore for example: '01679.DS17212.wig.gz'
    """
    global SMOOTHING
    if ignore_files is None:
        ignore_files = []
    epigenome_atlas = ftplib.FTP(host='ftp.genboree.org')
    epigenome_atlas.login()
    epigenome_atlas.cwd('EpigenomeAtlas/Current-Release/experiment-sample/Chromatin_Accessibility/')
    dirs = epigenome_atlas.nlst()
    print('Please select cell type:')
    print('-1', 'All')
    for i, d in enumerate(dirs):
        print(i, d)
    cell_type = input('Please enter number: ')
    pool_process = Pool()
    try:
        cell_type = int(cell_type)

        if cell_type >= len(dirs):
            raise ValueError()
        else:
            if cell_type == -1:
                sel_cell_types = list(dirs)
                sel_cell_types = sel_cell_types[1:]  # skip the meta dir
                sel_cell_types = sel_cell_types[26:]
            else:
                sel_cell_types = [dirs[cell_type]]
        sel_dir = ''
        try:
            for sel_dir in sel_cell_types:
                epigenome_atlas.cwd(sel_dir)
                print('cd: ', sel_dir)
                wig_files = [fl for fl in epigenome_atlas.nlst() if fl[-6:] == 'wig.gz']
                if cell_type > 0:
                    for i, fl in enumerate(wig_files):
                        print((i, fl))
                    selected_wig = input("Which file would you like to download? ")
                    selected_wigs = [wig_files[int(selected_wig)]]
                else:
                    selected_wigs = wig_files
                for selected_wig in selected_wigs:
                    if any(ig in selected_wig for ig in ignore_files):
                        print('Ignored:', selected_wig)
                        continue
                    if not os.path.exists(os.path.join(RAW_DATA_DIR, selected_wig)):
                        with open(os.path.join(DATA_DIR, selected_wig), 'wb') as dFile:
                            print(selected_wig)
                            epigenome_atlas.retrbinary('RETR %s' % selected_wig, dFile.write)
                        dFile.close()

                        print("%s download finished!" % selected_wig)

                        # create pickled small smoothed file
                        pool_process.apply_async(serialize_wig_file, (selected_wig,))
                    else:
                        print('Skipping - file already downloaded')
                if sel_dir != dirs[-1]:
                    epigenome_atlas.cwd('..')
                    time.sleep(3)  # sleep between directories moves
        except KeyboardInterrupt:
            print("KeyboardInterrupt: stopping downloading new files. Last dir: ", sel_dir)
        epigenome_atlas.close()
        pool_process.close()
        pool_process.join()

    except ValueError:
        print("The data you enter couldn't be parsed as index")


def download_ncbi_histone(markers_to_download=None, markers_to_ignore=None,
                          by_experiments_dir='pub/geo/DATA/roadmapepigenomics/by_experiment/'):
    """
    Downloads experiments results from NCBI.

    @param markers_to_download: specific experiments to be downloaded. Default: histone modifications+mRNA-Seq and RRBS
    @param markers_to_ignore: markers to ignore
    @param by_experiments_dir: NCBI directory for downloading experiments
    """
    if not markers_to_ignore:
        markers_to_ignore = ['DNase']
    import subprocess
    import time

    ncbi_ftp = ftplib.FTP(host='ftp.ncbi.nlm.nih.gov')
    ncbi_ftp.login()

    ncbi_ftp.cwd('/' + by_experiments_dir)
    if markers_to_download is None:
        experiments = ncbi_ftp.nlst('./')
        local_path = os.path.join(OTHER_DATA, "markers")
        if not os.path.exists(local_path):
            os.mkdir(local_path)
        markers_to_download = [ex for ex in experiments if (ex.startswith('H') or ex in ['mRNA-Seq', 'RRBS']) and not (
            os.path.exists(local_path + '/' + ex) and len(os.listdir(local_path + '/' + ex)) > 2)]
    enough_data = (ex for ex in markers_to_download if len(list(ncbi_ftp.nlst('./%s' % ex))) > 5)
    for ex in enough_data:
        print('Synchronizing %s' % ex)
        if any(ignore in ex for ignore in markers_to_ignore):
            continue

        ex_dir = by_experiments_dir + ex

        if os.path.exists(local_path + '/' + ex) and len(os.listdir(local_path + '/' + ex)) > 2:
            print('Skipping ex')
            continue

        subprocess.call(
            ["rsync", "-azuP", "--exclude=*.bed.gz", "--include=*.wig.gz", "ftp.ncbi.nlm.nih.gov::%s" % ex_dir,
             local_path])
        time.sleep(5)


def download_from_source(source_path, file_format="bigWig"):
    """
    Downloads based on a SOURCE file:
    *  each line in source contains a rsync directory
    * It looks for files.txt (if exist) to get metadata on the downloaded files
    @param file_format: file format to download
    @param source_path: a path to a SOURCE file to which data will be downloaded
    @return:
    """
    import subprocess
    import numpy as np
    import re

    with open(source_path, 'r') as source_file:
        sources = list(source_file.readlines())
    local_dir = os.path.dirname(source_path)

    meta_data_keys = ['file']
    meta_data = np.zeros((0, 1), dtype='S100')
    meta_file_path = os.path.join(local_dir, 'files.txt')
    for source in sources:
        source = source.strip()
        print('Download {} => {}'.format(source, local_dir))
        subprocess.call(
            ["rsync", "-azuP", "--include=*.{}".format(file_format), "--include=files.txt", "--exclude=*", source,
             local_dir])

        if not os.path.exists(meta_file_path):
            continue
        with open(meta_file_path, 'r') as meta_file:
            for track in meta_file.readlines():
                # skip non relevant files
                file_name, file_data = track.split('\t', 1)
                if not file_name.endswith('.' + file_format):
                    continue
                file_keys, file_values = zip(*re.findall('(.+?)=(.+?)[;\n$]', file_data))
                file_keys = [key.strip() for key in file_keys]
                new_meta_keys = [key for key in file_keys if key not in meta_data_keys]
                if any(new_meta_keys):
                    meta_data_tmp = meta_data
                    meta_data = np.zeros((meta_data.shape[0], meta_data.shape[1] + len(new_meta_keys)), dtype='S100')
                    meta_data[:, 0: meta_data_tmp.shape[1]] = meta_data_tmp
                meta_data_keys += new_meta_keys
                file_keys = map(lambda k: meta_data_keys.index(k), file_keys)
                new_row = np.zeros(meta_data.shape[1], dtype='S100')
                new_row[0] = file_name
                for meta_key, meta_value in zip(file_keys, file_values):
                    new_row[meta_key] = meta_value

                meta_data = np.vstack((meta_data, new_row))
        os.remove(meta_file_path)  # delete the meta file (avoid conflict with other sources)
    meta_data = np.vstack((meta_data_keys, meta_data))
    np.savetxt(os.path.join(local_dir, 'metadata.csv'), meta_data, delimiter='\t', fmt="%s")
    print('Consider to remove incorrect data! use the metadata.csv to find such data...')


def transform_ncbi(wig_directory=SIGNAL_DIR):
    """
    Transforms .wig.gz files in wig_directory to pkl files
    @param wig_directory: directory with cell types subdirectories, with wig files
    """
    pool_process = Pool()
    for cell in os.listdir(wig_directory):
        cell_path = os.path.join(wig_directory, cell)
        cell_files = os.listdir(cell_path)
        for f in cell_files:
            if not f.endswith('.wig.gz') or 'filtered-density' in f:
                continue
            output_file = f.replace('.gz', '').replace('.wig', '.%i.npz' % SMOOTHING)
            if output_file in cell_files:
                continue
            pool_process.apply_async(process_ncbi_file, (os.path.join(cell_path, f),))
    pool_process.close()
    pool_process.join()
    print('Finished transforming all files!')


def process_ncbi_file(wig_file):
    """
    pickle it
    @param wig_file: wiggle files to transform
    """
    print('Processing %s' % wig_file)
    SeqLoader.wig_transform(wig_file, SMOOTHING)
    print('end processing %s' % wig_file)


def transform_wig_files(directory=DATA_DIR):
    """
    Transforms wig.gz files to npz files and archives to RAW_DATA_DIR

    @param directory: directory with wig.gz files to transform
    """
    pool_process = Pool()
    for f in [f for f in os.listdir(directory) if f.endswith('.wig.gz')]:
        pool_process.apply_async(serialize_wig_file, (f, directory))
    pool_process.close()
    pool_process.join()


def serialize_wig_file(wig_file, directory=DATA_DIR):
    """
    serialize wig file to npz file
    @param directory: directory in which the wig file placed
    @param wig_file: wig file to npz/pickle
    """
    SeqLoader.wig_transform(os.path.join(directory, wig_file), SMOOTHING)
    print(os.path.join(directory, wig_file), '-->', os.path.join(RAW_DATA_DIR, wig_file))
    os.rename(os.path.join(directory, wig_file), os.path.join(RAW_DATA_DIR, wig_file))


def serialize_dir(in_directory=RAW_DATA_DIR, out_directory=SIGNAL_DIR, file_type='bigWig'):
    """
    Serialize bigwig file to npz file

    @param file_type: file types to serialize
    @param out_directory: output directory
    @param in_directory: input directory
    """
    import tempfile
    import subprocess

    if file_type == 'wig':
        return transform_wig_files()
    if file_type != 'bigWig':
        raise NotImplementedError

    for filename in os.listdir(in_directory):
        if not filename.endswith(file_type):
            continue
        src_file = os.path.join(in_directory, filename)
        dest_file = os.path.join(out_directory, filename.replace('.' + file_type, ''))
        if os.path.exists(dest_file+'.npz'):
            continue
        with tempfile.NamedTemporaryFile('w+', encoding='ascii') as tmp_file:
            subprocess.call([BIG_WIG_TO_BED_GRAPH, src_file, tmp_file.name])
            seq = SeqLoader.load_bg(tmp_file.name)
            SeqLoader.save_result_dict(dest_file, seq)
    print('Finish')


def wig_to_bed_graph(cur_trans):
    """
    Transforms wig file to bed graph file
    @param cur_trans: file to transform as 3-tuple (original.wig, temp.bw, result.bg))
    """
    import subprocess

    print('Transforming')
    print('->'.join(cur_trans))
    subprocess.call([WIG_TO_BIG_WIG, cur_trans[0], CHROM_SIZES, cur_trans[1]])
    subprocess.call([BIG_WIG_TO_BED_GRAPH, cur_trans[1], cur_trans[2]])
    os.remove(cur_trans[1])
    print('Completed')


def raw_data_to_bed_graph(wig_directory=RAW_DATA_DIR, bg_directory=BED_GRAPH_DIR):
    """
    Transforms raw data wig files to bed graph files
    @param wig_directory: directory with wig files
    @param bg_directory: directory with bed graph data
    """
    pool_process = Pool()
    bed_graphs = [f[:-3] for f in os.listdir(bg_directory)]
    need_transform = [(os.path.join(wig_directory, f), os.path.join(bg_directory, f[:-7] + '.bw'),
                       os.path.join(bg_directory, f[:-7] + '.bg')) for f in os.listdir(wig_directory) if
                      f[:-7] not in bed_graphs]
    for trans in need_transform:
        pool_process.apply_async(wig_to_bed_graph, (trans,))
    pool_process.close()
    pool_process.join()


def ucsc_download(src_path, target_path=None, email=None):
    """
    Downloads data from UCSC using FTP
    @param src_path: path to download to (local)
    @param target_path: path to download from (remote)
    @param email: email for authentication
    """
    if target_path is None:
        target_path = input("In which directory would you like to store the genome?")
    if email is None:
        email = input("Please enter your mail (will be used to enter to hgdownload ftp")
    with ftplib.FTP(host='hgdownload.cse.ucsc.edu') as ucsc_ftp:
        ucsc_ftp.login(user="anonymous", passwd=email)
        ucsc_ftp.cwd(os.path.dirname(target_path))
        if not os.path.exists(src_path):
            os.makedirs(src_path)

        with open(os.path.join(src_path, os.path.basename(target_path)), 'wb') as dFile:
            ucsc_ftp.retrbinary('RETR %s' % os.path.basename(target_path), dFile.write)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    subparsers = parser.add_subparsers(help="")
    # downloads genome sequence
    parser_download_genome = subparsers.add_parser('download_genome',
                                                   help='Downloads genome sequence from hgdownload.cse.ucsc.edu')
    parser_download_genome.add_argument('directory', help="Directory to store retrived file")
    parser_download_genome.add_argument('--genome', help="Genome to download", default='hg19')
    parser_download_genome.add_argument('--email', help="Email for authentication to UCSC", default='')
    parser_download_genome.set_defaults(
        func=lambda args: ucsc_download(args.directory, "goldenPath/%s/bigZips/%s.2bit" % (args.genome, args.genome),
                                        args.email))

    # utility function for downloading from multiple FTPs
    parser_download_source = subparsers.add_parser('download_sources',
                                                   help='Downloads genome sequence from hgdownload.cse.ucsc.edu')
    parser_download_source.add_argument('source',
                                        help="A file with each line containing FTP source to download data from")
    parser_download_source.set_defaults(func=lambda args: download_from_source(args.source))

    parser_transform_ncbi = subparsers.add_parser('transform_ncbi',
                                                  help='Transforms .wig.gz files in SIGNAL_DIR to pkl files')
    parser_transform_ncbi.add_argument('--directory', help="directory with cell types subdirectories, with wig files",
                                       default=SIGNAL_DIR)
    parser_transform_ncbi.set_defaults(func=lambda args: transform_ncbi(args.directory))

    parser_download_ncbi_markers = subparsers.add_parser('ncbiMarkers',
                                                         help='ownloads ncbi markers to OTHER_DATA/markers')
    parser_download_ncbi_markers.add_argument('--markers_to_download',
                                              help="specific experiments to be downloaded. " +
                                                   "Default: histone modifications+mRNA-Seq and RRBS",
                                              default=None)
    parser_download_ncbi_markers.add_argument('--markers_to_ignore', help="markers to ignore",
                                              default=None)
    parser_download_ncbi_markers.add_argument('--by_experiments_dir', help="NCBI directory for downloading experiments",
                                              default="pub/geo/DATA/roadmapepigenomics/by_experiment/")
    parser_download_ncbi_markers.set_defaults(
        func=lambda args: download_ncbi_histone(args.markers_to_download, args.markers_to_ignore,
                                                args.by_experiments_dir))

    raw_data_to_bed_graph_parser = subparsers.add_parser('raw_to_bed',
                                                         help='Transforms .wig.gz files in NCBI_DIR to pkl files')
    raw_data_to_bed_graph_parser.add_argument('--wig_directory', help="directory with wig files",
                                              default=RAW_DATA_DIR)
    raw_data_to_bed_graph_parser.add_argument('--bg_directory', help="directory with bed graph data",
                                              default=BED_GRAPH_DIR)
    raw_data_to_bed_graph_parser.set_defaults(func=lambda args: raw_data_to_bed_graph(args.wig_directory,
                                                                                      args.bg_directory))

    wig_to_npz_transform = subparsers.add_parser('wig_to_npz',
                                                 help='Transforms .wig.gz files in directory to npz files')
    wig_to_npz_transform.add_argument('--directory', help="directory with wig.gz files to transform",
                                      default=DATA_DIR)
    wig_to_npz_transform.set_defaults(func=lambda args: transform_wig_files(args.directory))

    serialize_dir_transform = subparsers.add_parser('serialize_dir',
                                                    help='Serializes wig.gz/bigWig files to npz')
    serialize_dir_transform.add_argument('--in_directory', help="Input directory", default=RAW_DATA_DIR)
    serialize_dir_transform.add_argument('--out_directory', help="Output directory directory", default=SIGNAL_DIR)
    serialize_dir_transform.set_defaults(func=lambda args: serialize_dir(args.in_directory, args.out_directory))

    command_args = parser.parse_args()
    command_args.func(command_args)
