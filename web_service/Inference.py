#!/usr/bin/python
#######
# Using ThermalGen to translate the normal proteins to thermal stable protein
# makrliou 2020/8/11
######
import tensorflow as tf
import tensorflow_addons as tfa
import numpy as np
import sequence_handler as sh

def Inference(seq, model_path="./models/nontherm2therm_G.h5", seq_len = 512):
    ThermalGen = tf.keras.models.load_model("./models/nontherm2therm_G.h5")
    #print(ThermalGen.summary())
    
    candidate_seq = sh.seq2index(seq)
    candidate_seq = tf.keras.preprocessing.sequence.pad_sequences([candidate_seq], value=0, maxlen=seq_len)
    candidate_seq = tf.one_hot(candidate_seq, depth=21)

    Therm_seq = sh.index2seq(tf.argmax(ThermalGen(candidate_seq), axis=-1))[0]

    return Therm_seq
pass

hemoglobin_seq = 'MHSSIVLATVLFVAIASASKTRELCMKSLEHAKVGTSKEAKQDGIDLYKHMFEHYPAMKKYFKHRENYTPADVQKDPFFIKQGQNILLACHVLCATYDDRETFDAYVGELMARHERDHVKVPNDVWNHFWEHFIEFLGSKTTLDEPTKHAWQEIGKEFSHEISHHGRHSVRDHCMNSLEYIAIGDKEHQKQNGIDLYKHMFEHYPHMRKAFKGRENFTKEDVQKDAFFVNKDTRFCWPFVCCDSSYDDEPTFDYFVDALMDRHIKDDIHLPQEQWHEFWKLFAEYLNEKSHQHLTEAEKHAWSTIGEDFAHEADKHAKAEKDHHEGEHKEEHH'

Therm_seq = Inference(hemoglobin_seq)
print(Therm_seq)


