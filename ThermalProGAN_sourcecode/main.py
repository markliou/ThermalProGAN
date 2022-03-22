import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'
os.environ['TF_XLA_FLAGS'] = '--tf_xla_cpu_global_jit'
os.environ['TF_FORCE_GPU_ALLOW_GROWTH'] = 'true'

import tensorflow as tf 
import numpy as np 
import sequence_handler 
import SUNI as NN


def main():
    therm_training_file = 'thermophilic.fasta.uc100'
    nontherm_training_file = 'nonthermophilic.fasta.uc100.uc20'
    seq_len = 512

    them_seq = sequence_handler.ReadFasta_Index(therm_training_file)
    them_seq = tf.keras.preprocessing.sequence.pad_sequences(them_seq, value=0, maxlen=seq_len)
    nonthem_seq = sequence_handler.ReadFasta_Index(nontherm_training_file)
    nonthem_seq = tf.keras.preprocessing.sequence.pad_sequences(nonthem_seq, value=0, maxlen=seq_len)
    
    them2nontherm = NN.SeqGen(seq_len)
    nontherm2them = NN.SeqGen(seq_len)
    them_D = NN.D(seq_len)
    nonthem_D = NN.D(seq_len)
    
    trainer = NN.trainer(them_seq, nonthem_seq, them2nontherm, nontherm2them, them_D, nonthem_D, seq_len, bs=32)
    trainer.triger()

    print('\n\nOptimization finish\n\n')

pass 

if __name__ == "__main__":
    main()
pass
