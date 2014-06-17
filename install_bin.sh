mkdir bin
cd bin
wget http://hgdownload.cse.ucsc.edu/admin/exe/linux.x86_64/bigWigToBedGraph
wget http://hgdownload.cse.ucsc.edu/admin/exe/linux.x86_64/wigToBigWig
wget http://hgdownload.cse.ucsc.edu/admin/exe/linux.x86_64/bedToBigBed
wget http://hgdownload.cse.ucsc.edu/admin/exe/linux.x86_64/bedGraphToBigWig
wget http://hgdownload.cse.ucsc.edu/admin/exe/linux.x86_64/bigBedToBed
wget http://hgdownload.cse.ucsc.edu/admin/exe/linux.x86_64/fetchChromSizes
chmod u+x bigWigToBedGraph wigToBigWig bedToBigBed bedGraphToBigWig bigBedToBed fetchChromSizes
fetchChromSizes hg19 > hg19.chrom.sizes


