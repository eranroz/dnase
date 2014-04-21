/**
 * @brief Simple parser for bed graph files
 * @author Eranroz
 * @date April 2014
 * @version 1.00
 */

#include <stdio.h>

class BgParser {

public:
	BgParser():m_fd(NULL){}
	bool init(const char* filename){
		m_fd = fopen(filename, "r");
		return m_fd != NULL;
	}
	bool read_next(char* chrom, int *start, int* end, float* score) { 
		return fscanf(m_fd, "%s\t%d\t%d\t%f",chrom, start, end, score) != EOF;
	}

	void close(){
		if (m_fd != NULL) {
			fclose(m_fd);
		}
	}

private:
	FILE *m_fd;
};