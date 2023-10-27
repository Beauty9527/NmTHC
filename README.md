# NMTHC
A hybrid error correction method based on a generative Neural Machine Translation model with transfer learning    
    
run the sh file "run_nmthc.sh"  to correct the long reads    
    
the environment requires tensorflow-gpu2.3 as the backends of keras 2.3, any version minimap2, samtools, python3.6    
    
make sure the samtools -mpileup command can generate the whole pileup file    

The size of GPU needed here is more than 28131MiB, if your GPU is smaller, try to decrease the batch_size from 64 to 32.    

