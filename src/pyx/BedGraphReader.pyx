# distutils: language = c++
# distutils: sources = BgParser.cpp
"""
Author: eranroz
"""
import numpy as np
cimport numpy as np
from libcpp.string cimport string
cimport cython
DTYPE = np.float
ctypedef np.float_t DTYPE_t

from libcpp cimport bool

VERBOSE = True

cdef extern from "BgParser.cpp":
	cdef cppclass BgParser:
		BgParser()
		bool init(const char* filename)
		bool read_next(char* chrom, int* a,int* b, float* c)
		void close()


def load_bedgraph(infile, chrom_size_file, bin_size=20):
	"""
	Loads bed graph. Bed graph must be with valid size chrom sizes and with no header
	"""
	return loadfile(infile, chrom_size_file)

@cython.boundscheck(False)
@cython.profile(False)
cdef loadfile(infile, dict chrom_sizes, int bin_size=20):
	cdef BgParser Parser
	cdef int start,end  # for each line
	cdef float score
	cdef float cache_score=0
	cdef int cache_start
	cdef int i=0  # temp variable
	cdef int curr_chrom_size = 0
	cdef bytes byt = infile.encode()
	cdef np.ndarray[DTYPE_t, ndim=1] chrom_data = np.zeros([1], dtype=DTYPE)
	
	cdef string curr_chrom
	if not Parser.init(byt):
		raise Exception("Error opening file")

	chrom_dict = dict()

	cdef bytes last_chrom = b""
	cdef char* curr_chromC="chr1X"
	try:
		while(Parser.read_next(curr_chromC, &start, &end, &score)):
			if last_chrom!=curr_chromC:
				if last_chrom==b"":
					last_chrom=curr_chromC
					curr_chrom_size = chrom_sizes[curr_chromC.decode('UTF-8')]
					chrom_data = np.zeros([curr_chrom_size], dtype=DTYPE)
					cache_start = curr_chrom_size  # invalidate
				else:
					if VERBOSE:
						print('Read %s'%last_chrom.decode('UTF-8'))
					if cache_start!=curr_chrom_size:
						chrom_data[cache_start] = cache_score
						cache_score = 0
					chrom_dict[last_chrom.decode('UTF-8')] = chrom_data
					last_chrom = curr_chromC
					curr_chrom_size = chrom_sizes[curr_chromC.decode('UTF-8')]
					chrom_data = np.zeros([curr_chrom_size], dtype=DTYPE)
					cache_start = curr_chrom_size  # invalidate

			start=start/bin_size
			end=end/bin_size
			if start>cache_start:
				chrom_data[cache_start] = cache_score
				cache_score = 0 
				cache_start = curr_chrom_size  # invalidate
			if end == start:
				cache_score += score
				cache_start = start

			if end > curr_chrom_size:
				raise IndexError("Invalid position encountered %i>%i for chrom %s" % (end*bin_size,curr_chrom_size, last_chrom) )
			for i in range(start, end):
				chrom_data[i] = score
		if cache_start != curr_chrom_size:
			chrom_data[cache_start] = cache_score
		chrom_dict[last_chrom.decode('UTF-8')] = chrom_data
	except:
		Parser.close()
		raise
	Parser.close()

	return chrom_dict


