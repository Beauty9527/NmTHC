#!/bin/bash -x
#
# This is a script to generate correction quality

REFERENCE_GENOME=$1
CORRECTED_LONG=$2
PLATFORM=$3

# Filter the reads by the length
seqtk seq -L 500 $2 > filter."$2"

# Corrected long reads are mapped to the reference using Minimap2

if [[ "${PLATFORM}" == "pb" ]]; then
    minimap2 -ax map-pb -L $1 filter."$2" | samtools view -bS | samtools sort -o "$2"_to_refer.sorted.bam
elif [[ "${PLATFORM}" == "ont" ]]; then
    minimap2 -ax map-ont -L $1 filter."$2" | samtools view -bS | samtools sort -o "$2"_to_refer.sorted.bam
else
    err "Invalid sequencing platform: ${PLATFORM}."
fi

# minimap2 -ax map-pb -L $1 filter."$2" | samtools view -bS | samtools sort -o "$2"_to_refer.sorted.bam
samtools index "$2"_to_refer.sorted.bam
rm filter."$2"

# We use samtools to generate stats
samtools stats -F 0x900 "$2"_to_refer.sorted.bam > "$2".stats
#samtools stats -F 0x900 long_to_refer.sorted.bam > 03.fasta.stats

# extract basic stats
grep ^SN "$2".stats | cut -f 2- > "$2".stats.txt 

# extract read length distribution (two columns, one for read length, # the other one for the number of reads in that length
grep ^RL "$2".stats | cut -f 2- > "$2".rl 

# run a python script to get N50
printf "N50:    " >> "$2".stats.txt 
python /itmslhppc/itmsl0212/validation/nmthc/N50.py -i "$2".rl --min-length 500 >> "$2".stats.txt 

# Calculate genome coverage
bedtools genomecov -max 1 -ibam "$2"_to_refer.sorted.bam > "$2".bed
printf "Genome fraction:    " >> "$2".stats.txt 
cut -f 5- "$2".bed | tail -1 >> "$2".stats.txt

rm "$2".rl
rm "$2".bed
rm "$2"_to_refer.sorted.*
rm "$2".stats


#How to use
#sh evaluate.sh NC_000913.3.fasta 03.fasta pb/ont
