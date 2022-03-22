import numpy as np
import tensorflow as tf 
import Bio.SeqIO 

ALPHA = '-ACDEFGHIKLMNPQRSTVWY'

def ReadFasta_Index(fasta_file):
    SEQs = []
    for rec in list(Bio.SeqIO.parse(fasta_file, 'fasta')):
        seq_ind = []
        if len(rec.seq) < 31:
            continue
        pass
        for res in rec.seq.upper():
            # print(tf.one_hot(ALPHA.find(res), 20))
            if (ALPHA.find(res)) != -1:
                seq_ind.append(ALPHA.find(res))
            else:
                #seq_ind.append('-')
                #break
                continue
            pass
        pass 

        SEQs.append(seq_ind)

    pass
    return(SEQs)
pass

def logit2seq(logits):
    seqs = []
    for seq in tf.argmax(logits, axis=-1).numpy():
        seqs.append(''.join(list(map(lambda i: ALPHA[i], seq))))
    pass
    return seqs
pass

def index2seq(seq_ind):
    seqs = []
    for seq in seq_ind.numpy():
        seqs.append(''.join(list(map(lambda i: ALPHA[i], seq))))
    pass
    return seqs
pass

def seq2index(seq):
    seq_ind = []
    for res in seq.upper():
        if (ALPHA.find(res)) != -1:
            seq_ind.append(ALPHA.find(res))
        else:
            continue
        pass
    pass
    return seq_ind
pass

def main():
    SEQs = ReadFasta_Index('ex.fasta')
    print(SEQs)
    SEQs = tf.keras.preprocessing.sequence.pad_sequences(SEQs, value=20, maxlen=3000)
    print('shape {}'.format(np.array(SEQs).shape))
    print(SEQs)
    SEQ_oh = tf.one_hot(SEQs, 21)
    print(SEQ_oh)
    SEQ_embeddings = tf.keras.layers.Embedding(input_dim=21, output_dim=512, mask_zero=True)(SEQs)
    print(SEQ_embeddings)

    # sess = tf.Session()
    # sess.run(tf.global_variables_initializer())
    # print(sess.run(SEQ_oh))
    # print(sess.run(SEQ_embeddings))
pass 

if __name__ == "__main__":
    main()
pass 
