"""
Classifier of genome based on DNase experiments.

This class also serve as a model manager, so you can persist a trained classifier in hard drive,
and store related files in same directory
"""
import logging
import os
import numpy as np
import pickle
from config import MODELS_DIR, DATA_DIR, MEAN_DNASE_DIR, PUBLISH_URL_PATH_MODELS
from data_provider import SeqLoader
from data_provider.LazyLoader import LazyChromosomeLoader
from data_provider.data_publisher import publish_dic
from data_provider import featureLoader
from hmm.bwiter import IteratorCondition
from html_utils import table_to_html_heatmap, list_to_ol

__author__ = 'eranroz'

_model_file_name = "model.pkl"


def model_dir(model_name):
    """
    Get the directory associated with model
    @param model_name: name of model
    @return: directory of the model
    """
    return os.path.join(MODELS_DIR, model_name)


def load(model_name):
    """
    Loads a model
    @rtype : DNaseClassifier
    @param model_name: name of model to be loaded
    @return: a segmentation model
    """
    model_path = os.path.join(model_dir(model_name), _model_file_name)
    if not os.path.exists(model_path):
        raise IOError("Model doesn't exist (%s)" % model_path)
    with open(model_path, 'rb') as f:
        return pickle.load(f)


def model_exist(model_name):
    """
    Check whether a model exist
    @param model_name: model name
    @return: true is such model exist otherwise false
    """
    return os.path.exists(os.path.join(model_dir(model_name), _model_file_name))


class ChromatinMetaClassifier(object):
    """
    Meta class for classifiers: mono classifier, multichannel classifier etc.
    """

    def __init__(self, strategy, resolution, name="unnamed"):
        """
        Initializes a new instance of C{Classifier}

        @type resolution: int
        @param strategy: strategy for classification
        @type strategy: C{ClassifierStrategy}
        """
        self.strategy = strategy
        self.name = name
        self.resolution = resolution
        if not isinstance(resolution, int):
            raise Exception('Resolution must be int')

    def fit(self, training_sequences):
        """
        Fits the model before actually running it
        @param training_sequences: training sequences for the Baum-Welch (EM)
        @return tuple(likelihood, fit_params)
        """
        p, fit_params = self.strategy.fit(training_sequences)
        return p, fit_params

    def classify(self, sequence_dicts):
        """
        Classifies each sequence
        @param sequence_dicts: sequences to classify
        @return: generator for classified sequences
        """
        for seq in sequence_dicts:
            print("classifying sequence")
            yield self.strategy.classify(seq)

    def __str__(self):
        strategy_str = str(self.strategy)
        return strategy_str

    def save(self, warn=True):
        """
        Save the classifier model in models directory

        @param warn: whether to warn when override a model
        """
        path = os.path.join(MODELS_DIR, self.name)
        if os.path.exists(path):
            if warn:
                logging.warning("Model already exist - overriding it")
                confirm = input("Do you want to override model \"%s\"? (Y/N)" % path).upper().strip()
                while confirm not in ['Y', 'N']:
                    confirm = input("Do you want to override model \"%s\"? (Y/N)" % path).upper().strip()

                if confirm == 'N':
                    return

        else:
            # create directory
            os.makedirs(path)

        with open(os.path.join(path, _model_file_name), 'wb') as model_file:
            pickle.dump(self, model_file)

    def model_dir(self, join=""):
        """
        Get the directory associated with this model

        @param join: name of file within model directory
        @return: directory name associated with this model
        """
        return os.path.join(model_dir(self.name), join)

    def segmentation_file_path(self):
        """
        Get the associated segmentation file path associated with the model.
        """
        return self.model_dir("segmentation.npz")


class DNaseClassifier(ChromatinMetaClassifier):
    """
    Classifier for active and non-active areas in chromatin based on DNase data for a single sequence
    """

    def fit_file(self, infile):
        """
        fit model based on input file
        @param infile: input file
        """
        transformer = self.strategy.data_transform()
        data = SeqLoader.load_dict(os.path.basename(infile), resolution=self.resolution, transform=transformer,
                                   directory=os.path.dirname(infile) or DATA_DIR)  # TODO: can load only partial
        self.fit([data])

    def classify_file(self, file_name, chromosome=None):
        """
        Loads data from file, transforms it and returns classification
        @param file_name: name of file
        @param chromosome: chromosome to classify or None for all chromosomes
        @return: a classification
        """
        transformer = self.strategy.data_transform()

        data = SeqLoader.load_dict(file_name, resolution=self.resolution, transform=transformer,
                                   directory=os.path.dirname(file_name) or DATA_DIR, chromosome=chromosome)
        return self.classify([data])

    def save_classify_file(self, file_name, out_file, save_raw=True, save_npz=True, save_bg=True):
        """
        Classifies file and saves it and related files to directory of the model
        @param file_name: name of file to classify
        @param out_file: name of output file without extension
        @param save_raw: whether to save raw data after transformation
        @param save_npz:  whether to save classified sequence as npz for later use
        @param save_bg: whether to save classified sequence as bg file (for UCSC)
        """
        transformer = self.strategy.data_transform()

        data = SeqLoader.load_dict(os.path.basename(file_name), resolution=self.resolution, transform=transformer,
                                   directory=os.path.dirname(file_name) or DATA_DIR)
        path_to_save = self.model_dir()
        if save_raw:
            print('Writing raw file')
            # SeqLoader.build_bedgraph(data, resolution=self.resolution,
            # output_file=os.path.join(path_to_save, '%s.raw.bg' % out_file))
            publish_dic(data,
                        self.resolution,
                        '%s.%s.raw' % (self.name, out_file),
                        short_label="Raw %s" % out_file,
                        long_label="Raw file after transformation")
        segmentation = self.classify([data])
        for classified_seq in segmentation:
            if save_bg:
                print('Writing result file (bg format)')
                publish_dic(classified_seq, self.resolution, '%s.%s' % (self.name, out_file),
                            short_label="%s-%s" % (out_file, self.name),
                            long_label="HMM classification.  %s" % (str(self)))
            if save_npz:
                print('Writing result file (npz format)')
                SeqLoader.save_result_dict(os.path.join(path_to_save, '%s.npz' % out_file), classified_seq)
        return segmentation

    def load_data(self, infile, chromosomes=None):
        """
        loads the data of file
        @param infile: file to load
        @param chromosomes: chromosomes to load
        """
        transformer = self.strategy.data_transform()
        data = SeqLoader.load_dict(infile, resolution=self.resolution, transform=transformer,
                                   directory=os.path.dirname(infile) or DATA_DIR, chromosome=chromosomes)
        return data


class DNaseMultiChannelClassifier(ChromatinMetaClassifier):
    """
    Classifier for multichannel

    strategy - should be C{GMMClassifier}
    """

    def __init__(self, strategy, resolution=1000, name=None):
        # assign default name for the model
        if name is None:
            name = 'multichannel%i' % resolution

        # whether to use sparse matrix representation during calculations
        # it seems that sparse matrix aren't required as num zeros is about 3% (chromosome 8, all cell types)
        # in resolution of 1000bp
        self.sparse = False
        super(DNaseMultiChannelClassifier, self).__init__(strategy, resolution, name)

    def _load_multichannel_data(self, directory=MEAN_DNASE_DIR, chromosomes=None):
        """
        Loads multichannel data. return a sparse representation
        @param directory: directory of files to analyze (directory with npz files)
        @param chromosomes: chromosomes to load
        @type directory: str
        @return: dict where keys are chromosomes and values are matrix: cell_types X genome position
        """
        return DNaseMultiChannelClassifier.load_multichannel_data(self.sparse, self.resolution, directory, chromosomes)

    @staticmethod
    def load_multichannel_data(sparse, resolution, directory=MEAN_DNASE_DIR, chromosomes=None):
        """
        Loads multichannel data. return a sparse representation
        @param resolution: resolution for binning
        @param sparse: whether to sue sparse representation
        @param directory: directory of files to analyze (directory with npz files)
        @param chromosomes: chromosomes to load
        @type directory: str
        @return: dict where keys are chromosomes and values are matrix: cell_types X genome position

        @remarks:
        """

        if sparse:
            import scipy.sparse

            compress = lambda x: scipy.sparse.coo_matrix(x)
            vstack = scipy.sparse.vstack
        else:
            compress = lambda x: x
            vstack = np.vstack
        all_cells = []
        cell_types_paths = [os.path.join(directory, cell_type) for cell_type in os.listdir(directory)]
        cell_types_paths.sort()  # deterministic...
        for cell_i, cell_type_path in enumerate(cell_types_paths):
            logging.info('Loading %s (%i/%i)' % (cell_type_path, cell_i + 1, len(cell_types_paths)))
            cell_data = SeqLoader.load_result_dict(cell_type_path)
            cell_data_new = dict()  # chromosome to down-sampled sparse matrix

            for k in (cell_data.keys() if chromosomes is None else chromosomes):
                cell_data_new[k] = compress(SeqLoader.down_sample(cell_data[k], resolution / 20))
            all_cells.append(cell_data_new)

        # organize be chromosomes
        if chromosomes is None:
            chromosomes = all_cells[0].keys()
        chromosomes_dic = dict()
        len_dim = 1 if sparse else 0
        for chromosome in chromosomes:

            max_length = max([cell[chromosome].shape[len_dim] for cell in all_cells])
            rows_matrices = []
            for cell in all_cells:
                if max_length > cell[chromosome].shape[len_dim]:
                    tmp = np.zeros(max_length)
                    if sparse:
                        tmp[0:cell[chromosome].shape[len_dim]] = cell[chromosome].todense()
                    else:
                        tmp[0:cell[chromosome].shape[len_dim]] = cell[chromosome]
                    rows_matrices.append(compress(tmp))
                else:
                    rows_matrices.append(cell[chromosome])
            # each row represent different cell, each column different position
            chromosome_matrix = vstack(rows_matrices)

            chromosomes_dic[chromosome] = chromosome_matrix
        return chromosomes_dic

    def load_data(self, directory=MEAN_DNASE_DIR, chromosomes=None):
        """
        Lazy loads data from directory
        @param chromosomes: chromosomes to load
        @param directory: directory of files to analyze (directory with npz files)
        @return: dict where keys are chromosomes and values are matrix: cell_types X genome position
        """
        transform = self.strategy.data_transform()
        loader = LazyChromosomeLoader(lambda x: transform(self._load_multichannel_data(chromosomes=[x],
                                                                                       directory=directory)[x]),
                                      chromosomes=chromosomes)
        return loader

    def classify_data(self, data):
        """
        Classifies each sequence
        @param data: sequences to classify
        @return: chromosome dictionary with classified data
        """
        return self.strategy.classify(data)

    def html_description(self, training_dir):
        """
        Get textual description of the model
        @param training_dir: directory used for training the model
        """
        training_files_raw = [cell_type.replace('.npz', '').replace('.mean', '') for cell_type in
                              os.listdir(training_dir)]
        training_files_raw.sort()  # determinism...
        training_files = list_to_ol(training_files_raw)

        pca = self.strategy.pca_reduction
        # currently assume GaussianHMM is the only valid strategy...

        pca_table = table_to_html_heatmap(pca.w.T, row_labels=training_files_raw)

        states_covars = [state[0][1] for state in self.strategy.model.emission.mean_vars]

        covariance_states = [table_to_html_heatmap(state, caption='State %i' % (state_i + 1))
                             for state_i, state in enumerate(states_covars)]
        covariance_states = '\n'.join(
            ['<div style="float:left;">%s</div>' % cov_table for cov_table in covariance_states])
        readme_content = """
Multivariate gaussian model on the PCA transformation of many cells data.<br/>
<b>Resolution:</b> {resolution}

<div style="float:right">
<h3>State transition</h3>
{html_state_transition}
</div>

<h3>States meaning</h3>
Recovery of states using inverse PCA may give us the following explanation:
{mean_states}

<h3> Model parameters </h3>
<ol>
<li> PCA - Data dimension is reduced from {num_training} to {pca_dims} using PCA.</li>
<li> HMM - Gaussian model (see below the learned parameters)</li>
</ol>

<h4>PCA</h4>
PCA base:<br>
{pca_table}

<h4>Training output</h4>
output parameters:
<pre style="background:#EEE;">
{strategy_str}
</pre>

<h4>EM converge</h4>
The following graph shows the convergence of the model is terms of log likelihood for the model describing the training
data compared to number of iterations:
<div style="text-align:center;">
<img src="{url_models_dir}/{model_name}/em-training.png"/>
</div>

Covariance matrices for different states:
<br style="clear:both;">
{covariance_states}

<br style="clear:both;">
Accessibility across cell types for different states:
<img src="{url_models_dir}/{model_name}/states_regions.png"/>

<h3>Training data</h3>
Using data from directory: {training_dir}

which includes:
{training_files}

""".format(**({
                                           'model_name': self.name,
                                           'url_models_dir': PUBLISH_URL_PATH_MODELS,
                                           'resolution': '%ibp' % self.resolution,
                                           'training_dir': training_dir,
                                           'training_files': training_files,
                                           'num_training': str(len(training_files_raw)),
                                           'pca_dims': str(pca.w.shape[0]),
                                           'strategy_str': str(self),
                                           'pca_table': pca_table,
                                           'mean_states': self.strategy.states_html(input_labels=training_files_raw),
                                           'html_state_transition': self.strategy.model.html_state_transition(),
                                           'covariance_states': covariance_states
                                       }))
        return readme_content

    def readme(self, training_dir, pca, likelihoods):
        """
        Creates an organized readme to describe the model and its training details
        @param likelihoods: array of likelihoods during fit
        @param pca: PCA matrix
        @param training_dir: directory used for training the model
        """
        import matplotlib

        matplotlib.use('Agg')
        from matplotlib import pyplot as plt
        # add readme
        training_files_raw = [cell_type.replace('.npz', '').replace('.mean', '') for cell_type in
                              os.listdir(training_dir)]
        training_files_raw.sort()  # determinism...
        training_files = ['\t %3i. %s' % (cell_i, cell_type.replace('.npz', '')) for cell_i, cell_type
                          in enumerate(training_files_raw)]

        # currently assume GaussianHMM is the only valid strategy...
        mean_vars_states = [state[0] for state in self.strategy.model.emission.mean_vars]
        mean_states = np.array([mean[0] for mean, var in mean_vars_states])
        mean_states = self.strategy.pca_reduction.recover(mean_states)
        readme_content = """
=Multichannel model=
===Training data===
Using data from directory: {training_dir}

which includes:
{training_files}
== Model parameters ==
1. PCA - Data dimension is reduced from {num_training} to {pca_dims} using PCA.
2. HMM - Gaussian model (see below the learned parameters)

===PCA===
{pca_str}

===Training output===
output parameters:
{strategy_str}

==States meaning==
Recovery of states using inverse PCA
{mean_states}
""".format(**({
                                           'training_dir': training_dir,
                                           'training_files': '\n'.join(training_files),
                                           'num_training': str(len(training_files)),
                                           'pca_dims': str(pca.w.shape[0]),
                                           'strategy_str': str(self),
                                           'pca_str': np.array_str(pca.w, precision=2, suppress_small=True,
                                                                   max_line_width=250).replace('\n\n', '\n'),
                                           'mean_states': np.array_str(mean_states, precision=2, suppress_small=True,
                                                                       max_line_width=250).replace('\n\n', '\n')
                                       }))
        with open(self.model_dir("readme.txt"), 'w') as readme:
            readme.write(readme_content)

        # some plots!
        # plot likelihood vs iterations (did we converge? probably yes...)
        plt.plot(likelihoods)
        plt.title('Likelihood for model')
        plt.ylabel('log likelihood')
        plt.xlabel('iterations')
        plt.savefig(self.model_dir('em-training.png'))

        # plot states
        plt.imshow(mean_states.T, cmap=plt.cm.Blues, extent=(0, mean_states.shape[0], 0, mean_states.shape[1]),
                   interpolation='none')
        plt.yticks(np.arange(len(training_files_raw)) + 0.5, training_files_raw, fontsize='x-small')  # , rotation=90
        plt.xlabel('states')
        plt.ylabel('cell type')
        plt.tight_layout()
        plt.savefig(self.model_dir("states.png"))


class MultiExpClassifier(ChromatinMetaClassifier):
    """
    A classifier of multiple experiments, such as different ChIP-Seqs, DNase etc.
    """

    def __init__(self, strategy, resolution=1000, name=None, experiments=None):
        # assign default name for the model
        if name is None:
            name = 'MultiExp%i%s' % (resolution, strategy.name())

        # whether to use sparse matrix representation during calculations
        # it seems that sparse matrix aren't required as num zeros is about 3% (chromosome 8, all cell types)
        # in resolution of 1000bp
        self.sparse = False
        self.experiments = experiments  # experiments used to define the features in the data
        self.train_chromosome = 'chr8'  # chromosome used for training
        super(MultiExpClassifier, self).__init__(strategy, resolution, name)

    def fit(self, data, stop_condition=IteratorCondition(10), filter_non_mappable=True):
        """
        Fits the model before actually running it
        @param filter_non_mappable: filter non mappable regions from the training
        @param stop_condition: converge condition to stop fitting (default: 10 iterations)
        @param data: training sequence for the Baum-Welch (EM)
        @return tuple(likelihood, fit_params)
        """
        training_seqs = data[self.train_chromosome]
        sequence_axis = np.argmax([training_seqs.shape[0], training_seqs.shape[1]])

        # break based on mappability
        if filter_non_mappable:
            min_run_size = 100
            min_mappability = 0.7
            mappability_orig_resolution = 20
            mappability_scale = self.resolution // mappability_orig_resolution
            mappability = featureLoader.load_mapability()[self.train_chromosome]
            mappability = SeqLoader.down_sample(mappability, mappability_scale) * (1.0/mappability_scale)
            mappability_slices = np.ma.clump_masked(np.ma.masked_greater_equal(mappability, min_mappability))
            if sequence_axis == 0:
                mappable_splits = [training_seqs[chuck, :] for chuck in mappability_slices if
                                   chuck.stop - chuck.start > min_run_size]
            else:
                mappable_splits = [training_seqs[:, chuck] for chuck in mappability_slices if
                                   chuck.stop - chuck.start > min_run_size]
            training_seqs = mappable_splits
        else:
            training_seqs = [training_seqs]

        p, fit_params = self.strategy.fit(training_seqs, stop_condition)
        return p, fit_params

    def html_description(self, training_files, features, data):
        """
        Get textual description of the model
        @param data: same of the data used for creating features
        @param features: name of the features used (order similar to order in data)
        @param training_files: files used for training the model
        """
        training_files.sort()  # determinism...
        training_files = list_to_ol(training_files)

        readme_content = """
Multinomial HMM strategy.<br/>
<b>Resolution:</b> {resolution}

<div style="float:right">
<h3>State transition</h3>
{html_state_transition}
</div>

<h3>States meaning</h3>
{mean_states}

<br style="clear:both;"/>
<img src="{url_models_dir}/{model_name}/tss states.png" />

<img src="{url_models_dir}/{model_name}/tes states.png" />
<!-- The strategy can represent the meaning based on parameters, image representation and so on -->

<h3>Pre-processing</h3>
{pre-processing_html}

<div style="text-align:center;">
States heatmap:<br>
<img src="{url_models_dir}/{model_name}/states_regions.png"/>
</div>

<div style="text-align:center;">
Number of states in genome:<br>
<img src="{url_models_dir}/{model_name}/states-genome-pie.png"/>
</div>

<h3> Model parameters </h3>
<ol>
<li> State transition parameters (see above)</li>
<li> (kernel) Emission parameters (see below)</li>
</ol>

<h4>Training output</h4>
output parameters:
<pre style="background:#EEE;">
{strategy_str}
</pre>

<h4>EM converge</h4>
The following graph shows the convergence of the model in terms of log likelihood for the model describing the training
data compared to number of iterations:
<div style="text-align:center;">
<img src="{url_models_dir}/{model_name}/em-training.png"/>
</div>

<h3>Training data</h3>
{training_data}
""".format(**({
                                           'model_name': self.name,
                                           'url_models_dir': PUBLISH_URL_PATH_MODELS,
                                           'resolution': '%ibp' % self.resolution,
                                           'pre-processing_html': self.strategy.preprocessing_html(data,
                                                                                                   input_labels=features,
                                                                                                   model_name=self.name),
                                           'strategy_str': str(self),
                                           'mean_states': self.strategy.states_html(input_labels=features),
                                           'html_state_transition': self.strategy.model.html_state_transition(),
                                           'training_data': training_files
                                       }))
        return readme_content

    def readme(self, training_files, likelihoods):
        """
        Creates an organized readme to describe the model and its training details
        @param likelihoods: array of likelihoods during fit
        @param training_files: files used to train the model
        """
        import matplotlib

        matplotlib.use('Agg')
        from matplotlib import pyplot as plt
        # add readme
        training_files.sort()  # determinism...
        training_files = ['\t %3i. %s' % (cell_i, cell_type.replace('.npz', '')) for cell_i, cell_type
                          in enumerate(training_files)]

        readme_content = """
=Multinomial model model=
===Training data===
{training_files}

== Model parameters ==

{strategy_str}

""".format(**({
                                           'training_files': '\n'.join(training_files),
                                           'strategy_str': str(self)
                                       }))
        with open(self.model_dir("readme.txt"), 'w') as readme:
            readme.write(readme_content)

        # some plots!
        # plot likelihood vs iterations (did we converge? probably yes...)
        plt.plot(likelihoods)
        plt.title('Likelihood for model')
        plt.ylabel('log likelihood')
        plt.xlabel('iterations')
        plt.savefig(self.model_dir('em-training.png'))

        # plot states
        # TODO: nice graphics...

    def classify_data(self, data):
        """
        Use the strategy to classify data
        @param data: data to be classified (dictionary like, keys are chromosome names, values are data array)
        @return:
        """
        return self.strategy.classify(data)