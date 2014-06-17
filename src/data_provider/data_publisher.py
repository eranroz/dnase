"""
Manages a hub for easier access in UCSC genome browser
"""
import os
from config import PUBLISH_URL_PATH, PUBLISH_DIR
from data_provider import SeqLoader

__author__ = 'eranroz'


def add_track(track_name, url_name, short_label, long_label, genome="hg19"):
    """
    Adds description for the new track to the track config file
    @param track_name: name of the track
    @param url_name: url name for the track
    @param short_label: short label for the track
    @param long_label: longer description for the trach
    @param genome: relevant genome for the track
    """
    tracks_db_path = os.path.join(PUBLISH_DIR, genome, "trackDb.txt")
    colors = url_name.endswith('.bb')
    if colors:
        auto_format = "itemRgb On"
    else:
        auto_format = "autoScale on"
    track_config = """

track {track_name}
bigDataUrl {url_name}
shortLabel {short_label}
longLabel {long_label}
type {type}
{autoformat}
""".format(**({'track_name': track_name.replace('[', '_').replace(']', '_'),
               'url_name': url_name,
               'short_label': short_label,
               'long_label': long_label,
               'type': 'bigBed 9' if url_name.endswith('.bb') else 'bigWig',
               'autoformat': auto_format
                                 }))


    with open(tracks_db_path, 'a') as tracks_file:
        tracks_file.write(track_config)


def publish_dic(dic_to_publish, resolution, name, short_label="", long_label="", genome="hg19", colors=False):
    """
    Publish dictionary: transforms it to big wig place it in PUBLISH_DIR
    @param colors: False for bigWig otuput (gray scale), True for bigBed colored
    @param long_label: longer description to explain in track
    @param genome: relevant genome from UCSC such as hg19
    @param short_label: description for the file track
    @param dic_to_publish: data to publish
    @param resolution: resolution of the data
    @param name: name for the file and the track
    """
    import tempfile

    with tempfile.NamedTemporaryFile('w+', encoding='ascii') as tmp_file:
        if name.endswith('.bw'):
            name = name[:-3]  # trim it
        if colors:
            SeqLoader.build_bed(dic_to_publish, resolution=resolution, output_file_name=tmp_file.name)
            SeqLoader.bed_to_bigbed(tmp_file.name, os.path.join(PUBLISH_DIR, genome, '%s.bb' % name))
            track_header = 'track type=bigBed name="%s" description="%s" bigDataUrl="%s/%s/%s.bb"'
        else:
            SeqLoader.build_bedgraph(dic_to_publish, resolution=resolution, output_file=tmp_file)
            SeqLoader.bg_to_bigwig(tmp_file.name, os.path.join(PUBLISH_DIR, genome, '%s.bw' % name))
            track_header = 'track type=bigWig name="%s" description="%s" bigDataUrl="%s/%s/%s.bw"'
    print(track_header % (name, short_label, PUBLISH_URL_PATH, genome, name))
    add_track(name.replace(' ', '_'), ('%s.bb' if colors else '%s.bw') % name, short_label or name, long_label,
              genome=genome)
    return track_header


def create_hub(short_label, long_label, email, genomes=['hg19']):
    # TODO: create genomes.txt and hub.txt
    """
    Creates a new hub directory
    @param genomes: genomes in your hub
    @param short_label: short label for the hub
    @param long_label: longer label for the hub
    @param email: your email
    """
    hub_description = """hub %{hub_name}
shortLabel {short_label}
longLabel {long_label}
genomesFile genomes.txt
email {email}
""".format(**(dict(short_label=short_label, long_label=long_label, email=email)))

    tracks_db_path = os.path.join(PUBLISH_DIR, "hub.txt")
    with open(tracks_db_path, 'w') as tracks_file:
        tracks_file.write(hub_description)

    with open(os.path.join(PUBLISH_DIR, "genomes.txt"), 'w') as genome_file:
        genome_file.write('\n\n'.join(['genome %s\ntrackDb %s/trackDb.txt' % g for g in genomes]))

    for g in genomes:
        os.makedirs(os.path.join(PUBLISH_DIR, g))


if __name__ == "__main__":
    import argparse
    import numpy as np

    parser = argparse.ArgumentParser()
    parser.add_argument('file', help="npz file to publish")
    parser.add_argument('--name', help="name of the published track")
    parser.add_argument('--short_label', help="short label for the published track")
    parser.add_argument('--long_label', help="long label (description) for the published track")
    parser.add_argument('--file_resolution', dtype=int, help="resolution of the npz file (bin sizes)", default=20)
    args = parser.parse_args()
    data = SeqLoader.load_result_dict(args.file)
    data2 = dict()
    for k, v in data.items():
        if k == 'chrM':
            continue
        v = np.array(v)
        las = np.where(v > 0)[0]
        data2[k] = v[:las[-1]]
    publish_dic(data2, args.file_resolution, args.name or os.path.basename(args.file),
                short_label=args.short_label or os.path.basename(args.file),
                long_label=args.long_label or os.path.basename(args.file))
