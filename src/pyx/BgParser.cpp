/**
 * @brief Simple parser for bed graph files
 * @author Eranroz
 * @date April 2014
 * @version 1.00
 */

#include <stdio.h>
#include <string.h>
class BgParser {

public:
	BgParser():m_fd(NULL){}
	bool init(const char* filename){
		m_fd = fopen(filename, "r");
		m_PrevChrom[0] = '\0';
		return m_fd != NULL;
	}

	/**
	Reads next line
	*/
	bool read_next(char* chrom, unsigned int *start, unsigned int* end, float* score, bool* newChrom) {
		bool isEof = fscanf(m_fd, "%s\t%d\t%d\t%f",chrom, start, end, score) != EOF;
		*newChrom = strcmp(chrom, m_PrevChrom)!=0;
		if(*newChrom){
			strcpy(m_PrevChrom, chrom);
		}
		return isEof;
	}

	void close(){
		if (m_fd != NULL) {
			fclose(m_fd);
		}
	}

private:
	FILE *m_fd;
	char m_PrevChrom[50];
};


class BedWriter {

public:
	BedWriter():m_fd(NULL){}
	bool init(const char* filename){
		m_fd = fopen(filename, "w");
		return m_fd != NULL;
	}

	/**
	Reads next line
	*/
	void write(char* chrom, int start, int end, int score, char* rgb) {
		// score used here as name
		fprintf(m_fd, "\n%s\t%d\t%d\t%d\t0\t.\t%d\t%d\t%s",chrom, start, end, score, start, end, rgb);
	}

	void close(){
		if (m_fd != NULL) {
			fclose(m_fd);
		}
	}

private:
	FILE *m_fd;
};
