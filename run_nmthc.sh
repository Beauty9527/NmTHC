# this file is a script to align the short to long  usage: long short ref  ref is used evaluate only


long_reads = $1
short_reads = $2
reference = $3
  
output_file="corrected_"$1""
# the finall output fasta file 

# align and generate a complete mpileup if the dumplicated reads occurs
# python /itmslhppc/itmsl0212/Projects/python/wrsCode/transfer2.0/dumplicated_remove.py --data_dir ./ --input_file $1 --output_file unique_"$1"
# minimap2 --split-prefix=tmp -ax sr -t 30 unique_"$1" $2 -a --secondary=no -o short2long.bam  # if there are dumplicated reads, replace the minimap2 with these two commands 


# align and generate a complete mpileup

minimap2 --split-prefix=tmp -ax sr -t 30 $1 $2 -a --secondary=no -o short2long.bam 
samtools sort short2long.bam -o short2long.sorted.bam
samtools index short2long.sorted.bam
samtools mpileup short2long.sorted.bam -a -s -f $1 -o mpileup_genome.pileup -a


# split the mpileup file

mkdir split_mpileup
python split_pileup3.0.py --data-dir ./ --whole-mpileup mpileup_genome.pileup --split-rate 1000 --output-dir split_mpileup/


# make a corpus 

#!/bin/bash

# travel the files in the dir
cd split_mpileup
for i in {0..999}; do   # this is related to the split-rate
    # 构建文件名
    filename="chunk_${i}.mpileup"

    # 检查文件是否存在
    if [ -e "$filename" ]; then
        echo "Processing: $filename"

        # 在这里执行你的操作，例如使用文件作为输入
	python ts_corpus2.0.py --data_dir ./ --input_file chunk_${i}.mpileup --uncovered_file un_reads.fasta --covered_long_regions c_long_regions.fasta --covered_label_regions c_label_regions.fasta --uncovered_long_regions un_long_regions.fasta --uncovered_label_regions un_label_regions.fasta --long_corpus long_corpus.txt --label_corpus label_corpus.txt
	python train_multi_gpu.py --data-dir ./ --long-corpus long_corpus.txt --label-corpus label_corpus.txt --check-point-path weight/cp.ckpt --model-hdf5-path weight/model.h5
	python predict_bi_lstm.py --gpu 0 --data-dir ./ --long-corpus long_corpus.txt --label-corpus label_corpus.txt --check-point-path weight/cp.ckpt --model-hdf5-path weight/model.h5 --output predict_${i}.fasta
	python merge.py --data-dir ./ --predict-fasta predict_${i}.fasta --un-label-region un_label_regions.fasta --output-file ${i}chunk.fasta
	###
	echo "finish the "${i}" corrected fasta" 
	cat un_reads.fasta ${i}chunk.fasta >> ../"$output_file"
	rm -rf label_corpus.txt long_corpus.txt un*  c_l* predict*   # dont remove the temp file during debug
	
    else
        echo "file $filename not here"
    fi
done

echo "the corrected fasta is "$output_file""
rm -rf *.mpileup
cd ..
sh evaluate.sh $3 "$output_file" ont  # choose pb/ont as you wish
rm -rf *.bam *.fai *.bai


