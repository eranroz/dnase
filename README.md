HMM for learning chromatin organization
=====
Markov Models for learning the chromatin organization.


# Features
* Discrete and continuous Hidden Markov Model
* Score functions to evaluate genome segmentation, based on different assumptions:
 * enrichment of chromatin modifications
 * consistency between different samples
 * gene breaks - number of genes that fall between domains
* Various script for download, transform and integration of data from different sources
* Performance and portability: works under Linux and Windows with 64-bit. The project includes some cython code for critical code paths: a parser for bed graph files (parsing ~500Mb file in ~10 sec) and some HMM algorithms (Viterbi)

# Installation and usage
The project is written in python 3, with some cython and c in critical code paths. It doesn't have UI but some command line tools.

Installation:
* Clone the repository
* copy config_default.py to config.py and define the directories to work with
* pyx - compile using the following command:
> python setup.py build_ext --inplace

* Fill directories with specific project data/external dependencies:
 * data - Create data directory with your data. Recommended public data repository: http://www.ncbi.nlm.nih.gov/epigenomics
 * bin - fill with UCSC programs such as bedToBigBed, bigWigToBedGraph and wigToBigWig. (can be downloaded for example from http://hgdownload.cse.ucsc.edu/admin/exe/ )
 * results - create a directory for storing results

Data directory as well as results directories may require large disk space, and you may find it convenient to store data or results directories in another drive (ln -s ...). See also [Installation FAQ](#installFaq)

## Usage
>~"Use the source, Luke" (Obi-Wan Kenobi)

You are welcome to use the source and extend it. Some common tasks are provided as command line tools:
* data_provider directory
 * dataDownloader - Script for downloading and transforming data.
 * createMeanMarkers - Script for averaging different samples from same experiment/same cell type
* dnase_classify - train and classify chromatin to regions of open and closed
# <a name="installFaq"></a>Installation FAQ
Here I keep some annoying problems I encountered and their solutions. You are welcome to suggest more :)

In general: since it handles huge files, it is intended to be used by 64-bit environment.
Using 32-bit environment may cause memory errors or at least slowness.
Validate your working environment is 64-bit (OS, python and for development also the IDE)

### Bin files

Bin files are UCSC programs, and are used for transforming files formats. The required programs can be downloaded using install.sh or manually from http://hgdownload.cse.ucsc.edu/admin/exe/

If you run into problems getting it work, use this checklist:
* "cannot execute binary file"
 * Is the program compatible with your system? (64-bit?) The install.sh downloads 64-bit version. Find 32-bit version of the programs if it doesn't work.
* "error while loading shared libraries: libssl.so.10: cannot open shared object file: No such file or directory"
 * Get libssl. With Debian linux (such as Ubuntu):
```bash
	sudo apt-get install libssl1.0.0 libssl-dev
	sudo ln -s /lib/x86_64-linux-gnu/libssl.so.1.0.0 /usr/lib/libssl.so.10
	sudo ln -s /lib/x86_64-linux-gnu/libcrypto.so.1.0.0 /usr/lib/libcrypto.so.10
```

### pyx
pyx directory contains cython code for optimization of critical code paths.

You can build it using the following:
> python setup.py build_ext --inplace

or under MS Windows:
> python.exe setup.py build_ext --inplace --compiler=msvc --plat-name=win-amd64
Tested with VS 2012: to use new version of msvc compiler you may have to modify msvc9compiler.py in distutils and specially get_build_version and PLAT_TO_VCVARS as described in http://www.xavierdupre.fr/blog/2013-07-07_nojs.html)

