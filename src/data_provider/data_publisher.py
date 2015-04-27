"""
Manages a hub for easier access in UCSC genome browser
"""
import os
from config import PUBLISH_URL_PATH_HUB, PUBLISH_DIR, TRACK_DESCRIPTION_TEMPALTE

__author__ = 'eranroz'


def add_track(track_name, url_name, short_label, long_label, description_html=None, genome="hg19"):
    """
    Adds description for the new track to the track config file
    @param description_html: url of html with description for the track
    @param track_name: name of the track
    @param url_name: url name for the track
    @param short_label: short label for the track
    @param long_label: longer description for the track
    @param genome: relevant genome for the track
    """
    tracks_db_path = os.path.join(PUBLISH_DIR, genome, "trackDb.txt")
    colors = url_name.endswith('.bb')
    auto_format = ""
    if colors:
        auto_format += "itemRgb On"
    else:
        auto_format += "autoScale on"

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

    if description_html is not None:
        # description path must be the save name as the track name
        description_path = os.path.join(PUBLISH_DIR, genome, track_name.replace('[', '_').replace(']', '_') + '.html')
        with open(description_path, 'w') as description_file:
            description_file.write(description_html)


def publish_dic(dic_to_publish, resolution, name, short_label="", long_label="", genome=None, description_html=None,
                colors=False):
    """
    Publish dictionary: transforms it to big wig place it in PUBLISH_DIR
    @param description_html: html data to describe the track
    @param colors: False for bigWig otuput (gray scale), True for bigBed colored
    @param long_label: longer description to explain in track
    @param genome: relevant genome from UCSC such as hg19
    @param short_label: description for the file track
    @param dic_to_publish: data to publish
    @param resolution: resolution of the data
    @param name: name for the file and the track
    """
    import tempfile
    from data_provider import SeqLoader
    from config import GENOME

    if genome is None:
        genome = GENOME
    if not os.path.exists(os.path.join(PUBLISH_DIR, genome)):
        raise Exception('No hub! Create hub manually or using create_hub method')

    # if this is a path to saved dictionary...
    if isinstance(dic_to_publish, str):
        dic_to_publish = SeqLoader.load_result_dict(dic_to_publish)

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
    print(track_header % (name, short_label, PUBLISH_URL_PATH_HUB, genome, name))
    add_track(name.replace(' ', '_'), ('%s.bb' if colors else '%s.bw') % name, short_label or name, long_label,
              genome=genome, description_html=description_html)

    return track_header


def create_hub(short_label, long_label, email, genomes=['hg19']):
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
    if not os.path.exists(PUBLISH_DIR):
        os.makedirs(PUBLISH_DIR)

    tracks_db_path = os.path.join(PUBLISH_DIR, "hub.txt")
    print('Creating hub.txt in {}'.format(tracks_db_path))
    with open(tracks_db_path, 'w') as tracks_file:
        tracks_file.write(hub_description)

    with open(os.path.join(PUBLISH_DIR, "genomes.txt"), 'w') as genome_file:
        genome_file.write('\n\n'.join(['genome %s\ntrackDb %s/trackDb.txt' % g for g in genomes]))

    for g in genomes:
        os.makedirs(os.path.join(PUBLISH_DIR, g))


def create_description_html(description, methods, verification, credits_details, references):
    """
    @param description: Description for a track
    @param methods: details of methods used in the track
    @param verification: verification details for the track
    @param credits_details: credit details for the track
    @param references: references
    @return: a partial html with the given details
    """
    with open(TRACK_DESCRIPTION_TEMPALTE, 'r') as template_file:
        template = template_file.read()
    template = template.format(**({'Description': description,
                                   'Methods': methods,
                                   'Verification': verification,
                                   'Credits': credits_details,
                                   'References': references
                                  }))

    return template


def create_composite_track(tracks, columns, directory, short_label_func, parent_track_name='rawMM9Data',
                           parent_short_label='mm9_ENCODE_raw'):
    """
    Creates composite trackDb file. WARNING: wasn't fully tested
    """
    from config import GENOME

    long_label = parent_short_label
    tracks = []
    track_item_format = """    track {}
bigDataUrl {}
parent {} off
shortLabel {}
longLabel {}
subGroups {}
metadata {}
"""
    if short_label_func is None:
        short_label_func = lambda data_item: '{}|{}|{}|{}|{}'.format(data_item['cell'], data_item['age'],
                                                                     data_item['sex'],
                                                                     data_item['treatment'], item_name)

    for data_item in tracks:
        item_name = data_item['file'].replace('.bigWig', '')
        item_parent = parent_track_name
        item_url = '{}/{}/{}'.format(PUBLISH_URL_PATH_HUB, GENOME, data_item['file'])
        item_short_label = short_label_func(data_item)
        item_long_label = item_name
        subgroup = ' '.join(['{}={}'.format(col, data_item[col].replace(' ', '_')) for col in columns])
        item_meta_data = ' '.join(['{}={}'.format(col, value.replace(' ', '_')) for col, value in data_item.items()])
        tracks.append(track_item_format.format(item_name, item_url, item_parent, item_short_label, item_long_label,
                                               subgroup, item_meta_data))
    iter_to_str = lambda iter_object: ' '.join(map(lambda it: '{}={}'.format(it, it), iter_object))
    tracks = '\n'.join(tracks)
    subgroupsValues = []
    for col_i, col in enumerate(columns):
        col_values = set([data_item[col] for data_item in tracks])
        col_values = iter_to_str(col_values)
        subgroupsValues.append('subGroup{} {} {} {}'.format(col_i, col, col.capitalize(), col_values))

    meta = """track {}
compositeTrack on
visibility dense
shortLabel {}
longLabel {}
{}
dimensions dimX=factor dimY=cellType dimA=lab
filterComposite dimA
dragAndDrop subTracks
type bigWig
autoscale off
sortOrder view=+ factor=+ cellType=+ lab=+
	track uniformDnaseSignal
	parent uniformDnase
	shortLabel ENCODE DNase-seq Signal
	view Signal
	visibility dense
{}
""".format(parent_track_name, parent_short_label, long_label, '\n'.join(subgroupsValues), tracks)

    tracks_db_path = os.path.join(PUBLISH_DIR, GENOME, "trackDb.txt")
    with open(tracks_db_path, 'w') as track_db:
        track_db.write(meta)
        track_db.close()


def composite_track_from_metadata(metadata_path, directory, parent_track_name='rawMM9Data',
                                  parent_short_label='mm9_ENCODE_raw'):
    """
     Creates composite trackDb file.
     """
    import csv

    short_label_func = lambda data_item: '{}|{}|{}|{}|{}'.format(data_item['cell'], data_item['age'], data_item['sex'],
                                                                 data_item['treatment'], item_name)
    columns = ['lab', 'cell']
    tracks = []

    with open(metadata_path, 'r') as metadata:
        metadata_csv = csv.DictReader(metadata, delimiter='\t')
        # extract the experiment type
        items = [item for item in metadata_csv]
        ex_type = lambda item: item['dataType'] if item['dataType'] == 'DnaseSeq' else item['antibody'].split('_')[0]

        for data_item in items:
            if not os.path.exists(os.path.join(directory, data_item['file'])):
                continue
            item_name = data_item['file'].replace('.bigWig', '')
            data_item['factor'] = ex_type(data_item)
            tracks.append(data_item)

    create_composite_track(tracks, columns, directory, short_label_func, parent_track_name=parent_track_name,
                           parent_short_label=parent_short_label)


def _main():
    import argparse

    parser = argparse.ArgumentParser()
    subparsers = parser.add_subparsers(help="")

    public_track_parser = subparsers.add_parser('publish_result_track',
                                                help='publish result dict to hub')

    public_track_parser.add_argument('file', help="npz file to publish")
    public_track_parser.add_argument('--name', help="name of the published track")
    public_track_parser.add_argument('--short_label', help="short label for the published track")
    public_track_parser.add_argument('--long_label', help="long label (description) for the published track")
    public_track_parser.add_argument('--file_resolution', dtype=int, help="resolution of the npz file (bin sizes)",
                                     default=20)
    public_track_parser.set_defaults(
        func=lambda args: publish_dic(args.file, args.file_resolution,
                                      args.name or os.path.basename(args.file),
                                      short_label=args.short_label or os.path.basename(args.file),
                                      long_label=args.long_label or os.path.basename(args.file)))

    create_hub_parser = subparsers.add_parser('create_hub',
                                              help='Creates hub meta files/directories')
    create_hub_parser.add_argument('--short_label', help="short label for the hub", required=True)
    create_hub_parser.add_argument('--long_label', help="long label (description) for the hub", required=True)
    create_hub_parser.add_argument('--email', help="Email of the hub owner", required=True)
    create_hub_parser.add_argument('--genome', help="relevant genome for the track", required=True)
    create_hub_parser.set_defaults(
        func=lambda args: create_hub(args.short_label, args.long_label, args.email, genomes=[args.genome]))

    command_args = parser.parse_args()
    command_args.func(command_args)


if __name__ == "__main__":
    _main()
