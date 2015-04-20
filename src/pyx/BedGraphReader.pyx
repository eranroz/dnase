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
I_DTYPE = np.int
ctypedef np.int_t I_DTYPE_t

from libcpp cimport bool

VERBOSE = True

cdef extern from "BgParser.cpp":
	cdef cppclass BgParser:
		BgParser()
		bool init(const char* filename)
		bool read_next(char* chrom, unsigned int* a, unsigned int* b, float* c, bool* is_new_chrom)
		void close()
	cdef cppclass BedWriter:
		BedWriter()
		bool init(const char* filename)
		void write(char* chrom, int start, int end, int score, char* rgb)
		void close()

def load_bedgraph(infile, chrom_size_file, bin_size=20):
	"""
	Loads bed graph. Bed graph must be with valid size chrom sizes and with no header
	"""
	return loadfile(infile, chrom_size_file, bin_size)

def write_bed(chromDict, bin_size not None, outfile, color_schema=None):
	"""
	Write a bed file using chromDict as data
	"""

	if color_schema is None:
		import matplotlib as mpl
		import matplotlib.cm as cm
		max_v = np.max([np.max(v) for v in chromDict.values()])
		norm = mpl.colors.Normalize(vmin=0, vmax=max_v+1)
		cmap = cm.spectral
		m = cm.ScalarMappable(norm=norm, cmap=cmap)
		color_schema = dict()
		for i in range(0, max_v+1):
			rgb = list(m.to_rgba(i)[:3])
			for j in range(0,3):
				rgb[j] = str("%i"%(255*rgb[j]))
			color_schema[i] = ','.join(rgb).encode()
	return _write_bed(chromDict, bin_size, outfile, color_schema)

@cython.boundscheck(False)
@cython.profile(False)
@cython.cdivision(True)
cdef loadfile(infile, dict chrom_sizes, unsigned int bin_size=20):
	cdef BgParser Parser
	cdef unsigned int start,end  # for each line
	cdef float score
	cdef float cache_score=0
	cdef unsigned int cache_start, end_cache
	cdef unsigned int start_bin_size, cache_start_bin_size
	cdef unsigned int i=0  # temp variable
	cdef unsigned int curr_chrom_size = 0
	cdef bytes file_name = infile.encode()
	cdef np.ndarray[DTYPE_t, ndim=1] chrom_data = np.zeros([1], dtype=DTYPE)
	
	cdef string curr_chrom
	if not Parser.init(file_name):
		raise Exception("Error opening file")

	chrom_dict = dict()

	cdef bytes last_chrom = b""
	cdef char* curr_chromC="chr1X"
	cdef bool newChrom
	try:
		while(Parser.read_next(curr_chromC, &start, &end, &score, &newChrom)):
			#if last_chrom!=curr_chromC:
			if newChrom:
				if last_chrom!=b"":
					if VERBOSE:
						print('Read %s'%last_chrom.decode('UTF-8'))
					if cache_score!=0:
						if cache_start_bin_size==chrom_data.shape[0]:
							chrom_data[cache_start_bin_size-1] += cache_score/bin_size
						else:
							chrom_data[cache_start_bin_size] = cache_score/bin_size
						cache_score = 0
					chrom_dict[last_chrom.decode('UTF-8')] = chrom_data
				last_chrom = curr_chromC
				
				curr_chrom_size = chrom_sizes[curr_chromC.decode('UTF-8')]
				chrom_data = np.zeros([curr_chrom_size], dtype=DTYPE)
				curr_chrom_size *= bin_size
				curr_chrom_size += bin_size # give extra bin size to avoid overflow
				cache_start = 0  # invalidate
				cache_start_bin_size = 0
				cache_score = 0

			# check if cache maintance required
			if cache_score:
				# place the cache in chrom_data, or update it
				if (start/bin_size)!=cache_start_bin_size:
					chrom_data[cache_start_bin_size] = cache_score/bin_size
					cache_score = 0
				else:
					if end/bin_size == start/bin_size:
						cache_score += (end-start)
					else:
						end_cache = start-start%bin_size+bin_size
						chrom_data[cache_start_bin_size] = (cache_score+score*float(end_cache-start))/bin_size
						cache_score = 0
						start = end_cache
			
			if (end-start) >= bin_size:
				if cache_score:
					chrom_data[start/bin_size] = cache_score/bin_size
					cache_score = 0
				else:
					chrom_data[start/bin_size] = score
					
				for i in range((start/bin_size)+1, end/bin_size):
					chrom_data[i] = score

				cache_score = score*(end%bin_size)
				cache_start = end-end%bin_size
				cache_start_bin_size = cache_start/bin_size
			else:
				if cache_score == 0:
					cache_score = score*(end-start) # score*num of bp in cache
					cache_start = start
					cache_start_bin_size = cache_start/bin_size
			if end > curr_chrom_size:
				raise IndexError("Invalid position encountered %i>%i for chrom %s" % (end,curr_chrom_size, last_chrom) )

		if cache_score:
			chrom_data[cache_start_bin_size] = cache_score
		chrom_dict[last_chrom.decode('UTF-8')] = chrom_data
	except:
		Parser.close()
		raise
	Parser.close()

	return chrom_dict


@cython.boundscheck(False)
@cython.profile(False)
cdef _write_bed(dict chromDict, int bin_size, str outfile, dict color_schema):
	cdef BedWriter BgWriter
	cdef int start  # for each line
	cdef int score
	cdef bytes outfile_byt = outfile.encode()
	cdef int prev_v
	cdef int pos
	cdef int chrom_size
	cdef bytes chromName_byt
	cdef bytes rgb_v
	cdef string curr_chrom
	cdef np.ndarray[I_DTYPE_t, ndim=1] chrom_data
	if not BgWriter.init(outfile_byt):
		raise Exception("Error creating file")
	score = 0

	for chromName, chromValue in chromDict.items():
		chrom_data = np.array(chromValue,dtype=I_DTYPE)
		chrom_size = chrom_data.shape[0]
		chromName_byt = chromName.encode()
		start=0
		pos = 0
		prev_v = chrom_data[0]
		#print(chromValue)
		#for v in chrom_data:
		for pos in range(chrom_size):
			v = chrom_data[pos]
			#if prev_v==-1:
			#	prev_v = v
			if v!=prev_v:
				rgb_v = color_schema[prev_v]
				BgWriter.write(chromName_byt, start*bin_size, (pos)*bin_size, prev_v, rgb_v)
				start = pos
				prev_v = v
		print('Writed %s'%chromName)
		rgb_v = color_schema[prev_v]
		BgWriter.write(chromName_byt, start*bin_size, pos*bin_size, score, rgb_v)
	BgWriter.close()
