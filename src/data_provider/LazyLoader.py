import numpy as np
from config import GENOME
__author__ = 'eranroz'


class LazyChromosomeLoader(object):
    """
    Dictionary like object for lazy load of data:
    keys are chromosomes and values are the data for those chromosomes

    the data is loaded only once the user access the chromosomes
    """
    def __init__(self, loader, chromosomes=None):
        self.loader = loader
        if chromosomes is None:
            if GENOME == 'hg19':
                self.chromosomes = ['chr%i' % i for i in np.arange(1, 23)]
                self.chromosomes += ['chrX', 'chrY']
            elif GENOME == 'mm9':
                self.chromosomes = ['chr%i' % i for i in np.arange(1, 20)]
                self.chromosomes += ['chrX', 'chrY']
        else:
            self.chromosomes = chromosomes

    def items(self):
        """
        Get items in lazy loader e.g. chromosomes

        """
        for chrom in self.chromosomes:
            yield chrom, self.loader(chrom)

    def __getitem__(self, x):
        return self.loader(x)