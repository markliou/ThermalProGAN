import tensorflow as tf 
import tensorflow_addons as tfa
import numpy as np 
import sequence_handler as sh
import os
import multiprocessing as mp
import threading

# XLA accelerating
tf.keras.backend.clear_session()
tf.config.optimizer.set_jit(True) # Enable XLA.
tf.autograph.set_verbosity(2)

def SeqGen(seq_len=512):
    # input shape: [N, 2000]
    # outpu shape: [N, 2000, 512]
    
    REG = tf.keras.regularizers.L2(l2=1e-4)
    
    input_seq = tf.keras.layers.Input(shape=[seq_len, 21]) 
    seq_embeddings = tf.keras.layers.Conv1D(filters=128, kernel_size=1, strides=1, padding='SAME', activation=None, activity_regularizer=REG)(input_seq)
    pos_emb_v_in = tf.random.normal([seq_len, 128], stddev=.05)
    seq_position_encoding = seq_embeddings + pos_emb_v_in #position embedding

    seq_encoding = tf.keras.layers.LayerNormalization(axis=-2)(seq_position_encoding)
    en1 = tf.keras.layers.Conv1D(filters=128, kernel_size=9, strides=2, padding='SAME', activation=tfa.activations.mish, kernel_initializer='he_uniform', kernel_regularizer=REG, activity_regularizer=None)(seq_encoding) #1000, 128
    en1n = tf.keras.layers.LayerNormalization(axis=-2)(en1)
    en2 = tf.keras.layers.Conv1D(filters=256, kernel_size=9, strides=2, padding='SAME', activation=tfa.activations.mish, kernel_initializer='he_uniform', kernel_regularizer=REG, activity_regularizer=None)(en1n) #500, 256
    en2n = tf.keras.layers.LayerNormalization(axis=-2)(en2)
    en3 = tf.keras.layers.Conv1D(filters=512, kernel_size=9, strides=2, padding='SAME', activation=tfa.activations.mish, kernel_initializer='he_uniform', kernel_regularizer=REG, activity_regularizer=None)(en2n) #250, 512
    en3n = tf.keras.layers.LayerNormalization(axis=-2)(en3)

    latent1_s = tf.keras.layers.Conv1D( filters=512, kernel_size=9, strides=1, padding='SAME', activation=None, kernel_initializer='he_uniform', kernel_regularizer=REG, activity_regularizer=None)(en3n)
    latent1_g = tf.keras.layers.Conv1D( filters=512, kernel_size=9, strides=1, padding='SAME', activation=tf.nn.sigmoid, kernel_initializer='he_uniform', kernel_regularizer=REG, activity_regularizer=None)(en3n)
    latent2_s = tf.keras.layers.Conv1D( filters=512, kernel_size=9, strides=1, padding='SAME', dilation_rate=2, activation=None, kernel_initializer='he_uniform', kernel_regularizer=REG, activity_regularizer=None)(en3n)
    latent2_g = tf.keras.layers.Conv1D( filters=512, kernel_size=9, strides=1, padding='SAME', dilation_rate=2, activation=tf.nn.sigmoid, kernel_initializer='he_uniform', kernel_regularizer=REG, activity_regularizer=None)(en3n)
    latent3_s = tf.keras.layers.Conv1D( filters=512, kernel_size=9, strides=1, padding='SAME', dilation_rate=3, activation=None, kernel_initializer='he_uniform' ,kernel_regularizer=REG, activity_regularizer=None)(en3n)
    latent3_g = tf.keras.layers.Conv1D( filters=512, kernel_size=9, strides=1, padding='SAME', dilation_rate=3, activation=tf.nn.sigmoid, kernel_initializer='he_uniform' ,kernel_regularizer=REG, activity_regularizer=None)(en3n)
    latent = (latent1_s * latent1_g + latent2_s * latent2_g + latent3_s * latent3_g)

    de1u = tf.keras.layers.UpSampling1D(size=2)(latent)
    de1u = tf.keras.layers.LayerNormalization(axis=-2)(de1u)
    de1g = tf.keras.layers.Conv1D(filters=256, kernel_size=9, strides=1, padding='SAME', activation=tfa.activations.mish, kernel_initializer='he_uniform', kernel_regularizer=REG, activity_regularizer=None)(de1u) + en2 #500, 256
    de2u = tf.keras.layers.UpSampling1D(size=2)(de1g) 
    de2u = tf.keras.layers.LayerNormalization(axis=-2)(de2u) 
    de2g = tf.keras.layers.Conv1D(filters=128, kernel_size=9, strides=1, padding='SAME', activation=tfa.activations.mish, kernel_initializer='he_uniform', kernel_regularizer=REG, activity_regularizer=None)(de2u) + en1 #1000, 128
    de3u = tf.keras.layers.UpSampling1D(size=2)(de2g)
    de3u = tf.keras.layers.LayerNormalization(axis=-2)(de3u)
    de4g = tf.keras.layers.Conv1D(filters=128, kernel_size=9, strides=1, padding='SAME', activation=tfa.activations.mish, kernel_initializer='he_uniform', kernel_regularizer=REG, activity_regularizer=None)(de3u) #2000, 128
    de4  = tf.keras.layers.LayerNormalization(axis=-2)(de4g)
   
    out_embs = tf.keras.layers.Conv1D(filters=128, kernel_size=9, strides=1, padding='SAME', activation=None, kernel_initializer='lecun_uniform', kernel_regularizer=REG, activity_regularizer=None)(de4)
    out_embg = tf.keras.layers.Conv1D(filters=128, kernel_size=9, strides=1, padding='SAME', activation=tf.nn.sigmoid, kernel_initializer='lecun_uniform', kernel_regularizer=REG, activity_regularizer=None)(de4)
    out_embn = out_embs * out_embg
    
    out_emb = out_embn

    out = tf.keras.layers.Conv1D(filters=21, kernel_size=1, strides=1, padding='SAME', activation=None, kernel_initializer='lecun_uniform', kernel_regularizer=REG, activity_regularizer=REG)(out_emb)
    
    seq_gen_model = tf.keras.Model(inputs=input_seq, outputs=out)

    def pos_regulizer(): # position embedding regularizer
        return tf.reduce_mean(REG(pos_emb_v_in))
    pass

    seq_gen_model.add_loss(pos_regulizer) # append the possition embedding regularizer in to model loss

    return seq_gen_model 
pass 


def D(seq_len=512):
    REG = tf.keras.regularizers.L2(l2=1e-3)
    
    input_seq = tf.keras.layers.Input(shape=[seq_len, 21])
    seq_embeddings = tf.keras.layers.Conv1D(filters=64, kernel_size=1, strides=1, padding='SAME', activation=None, kernel_regularizer=REG)(input_seq)
    pos_emb_v_in = tf.random.normal([seq_len, 64], stddev=.05)
    seq_embeddings += pos_emb_v_in #position embedding

    en1 = tf.keras.layers.Conv1D(filters=32, kernel_size=3, strides=2, padding='SAME', activation=tfa.activations.mish, kernel_regularizer=REG)(seq_embeddings) #len:256
    Tensor_pool = en1
    for smoothing_time in range(3):
        Tensor_pool += tf.keras.layers.BatchNormalization()(
                       tf.keras.layers.Conv1D(filters=32, kernel_size=3, dilation_rate=(smoothing_time+1), strides=1, padding='SAME', activation=tfa.activations.mish, kernel_initializer='he_uniform', kernel_regularizer=REG, activity_regularizer=None)(en1)
                       , training=True)
    pass
    Tensor_pool = tf.keras.layers.BatchNormalization()(Tensor_pool, training=True)
    en2 = tf.keras.layers.Conv1D(filters=32, kernel_size=3, strides=2, padding='SAME', activation=tfa.activations.mish, kernel_regularizer=REG)(Tensor_pool) #len:128
    Tensor_pool = en2
    for smoothing_time in range(3):
        Tensor_pool += tf.keras.layers.BatchNormalization()(
                       tf.keras.layers.Conv1D(filters=32, kernel_size=3, dilation_rate=(smoothing_time+1), strides=1, padding='SAME', activation=tfa.activations.mish, kernel_initializer='he_uniform', kernel_regularizer=REG, activity_regularizer=None)(en2)
                       , training=True)
    pass
    Tensor_pool = tf.keras.layers.BatchNormalization()(Tensor_pool, training=True)
    en3 = tf.keras.layers.Conv1D( filters=32, kernel_size=3, strides=2, padding='SAME', activation=tfa.activations.mish, kernel_regularizer=REG)(Tensor_pool) #len:64
    Tensor_pool = en3
    for smoothing_time in range(3):
        Tensor_pool += tf.keras.layers.BatchNormalization()(
                       tf.keras.layers.Conv1D(filters=32, kernel_size=3, dilation_rate=(smoothing_time+1), strides=1, padding='SAME', activation=tfa.activations.mish, kernel_initializer='he_uniform', kernel_regularizer=REG, activity_regularizer=None)(en3)
                       , training=True)
    pass
    
    Tensor_pool = tf.keras.layers.BatchNormalization()(Tensor_pool, training=True)
    en4 = tf.keras.layers.Conv1D( filters=64, kernel_size=3, strides=2, padding='SAME', activation=tfa.activations.mish, kernel_regularizer=REG)(Tensor_pool) #len:32
    Tensor_pool = en4
    for smoothing_time in range(3):
        Tensor_pool += tf.keras.layers.BatchNormalization()(
                       tf.keras.layers.Conv1D(filters=64, kernel_size=3, dilation_rate=(smoothing_time+1), strides=1, padding='SAME', activation=tfa.activations.mish, kernel_initializer='he_uniform', kernel_regularizer=REG, activity_regularizer=None)(en4)
                       , training=True)
    pass

    Tensor_pool = tf.keras.layers.BatchNormalization()(Tensor_pool, training=True)
    en5 = tf.keras.layers.Conv1D( filters=128, kernel_size=3, strides=2, padding='SAME', activation=tfa.activations.mish, kernel_regularizer=REG)(Tensor_pool) #len:16
    Tensor_pool = en5
    for smoothing_time in range(3):
        Tensor_pool += tf.keras.layers.BatchNormalization()(
                       tf.keras.layers.Conv1D(filters=128, kernel_size=3, dilation_rate=(smoothing_time+1), strides=1, padding='SAME', activation=tfa.activations.mish, kernel_initializer='he_uniform', kernel_regularizer=REG, activity_regularizer=None)(en5)
                       , training=True)
    pass


    Tensor_pool = tf.keras.layers.BatchNormalization()(Tensor_pool, training=True)
    en6 = tf.keras.layers.Conv1D( filters=256, kernel_size=3, strides=2, padding='SAME', activation=tfa.activations.mish, kernel_regularizer=REG)(Tensor_pool) #len:8
    Tensor_pool = en6
    for smoothing_time in range(3):
        Tensor_pool += tf.keras.layers.BatchNormalization()(
                       tf.keras.layers.Conv1D(filters=256, kernel_size=3, dilation_rate=(smoothing_time+1), strides=1, padding='SAME', activation=tfa.activations.mish, kernel_initializer='he_uniform', kernel_regularizer=REG, activity_regularizer=None)(en6)
                       , training=True)
    pass

    Tensor_pool = tf.keras.layers.BatchNormalization()(Tensor_pool, training=True)
    en7 = tf.keras.layers.Conv1D( filters=256, kernel_size=3, strides=2, padding='SAME', activation=tfa.activations.mish, kernel_regularizer=REG)(Tensor_pool) #len:4
    Tensor_pool = en7
    for smoothing_time in range(3):
        Tensor_pool += tf.keras.layers.BatchNormalization()(
                       tf.keras.layers.Conv1D(filters=256, kernel_size=3, dilation_rate=(smoothing_time+1), strides=1, padding='SAME', activation=tfa.activations.mish, kernel_initializer='he_uniform', kernel_regularizer=REG, activity_regularizer=None)(en7)
                       , training=True)
    pass

    # liner projection and pooling
    Tensor_pool = tf.keras.layers.BatchNormalization()(Tensor_pool, training=True)
    enp = tf.keras.layers.MaxPool1D(pool_size=4)(Tensor_pool)
    en7 = tf.keras.layers.Conv1D( filters=1, kernel_size=1, strides=1, padding='SAME', activation=None, kernel_regularizer=REG)(enp) #len:1

    out = tf.keras.layers.Flatten()(en7)
    
    D = tf.keras.Model(inputs=input_seq, outputs=out)

    def pos_regulizer(): # position embedding regularizer
        return tf.reduce_mean(REG(pos_emb_v_in))
    pass

    D.add_loss(pos_regulizer) # append the possition embedding regularizer in to model loss

    return D 
pass

class trainer():
    def __init__(self, tr_a, tr_b, seqgen_a2b, seqgen_b2a, da, db, seq_len, bs = 1):
        os.environ['TF_FORCE_GPU_ALLOW_GROWTH'] = 'true'
        
        self.therm_seq = tr_a 
        self.nontherm_seq = tr_b
        self.seq_len = seq_len

        self.therm2nontherm_G = seqgen_a2b
        self.nontherm2therm_G = seqgen_b2a
        self.therm_D = da
        self.nontherm_D = db
        
        self.iteration_no = 50000
        self.warmup_iteration_no = 50
        self.iter_counter = 0

        self.D_LEARNING_RATE = tf.keras.optimizers.schedules.PolynomialDecay(1E-4, 5000, 1E-5, 1.5)
        self.G_LEARNING_RATE = tf.keras.optimizers.schedules.PolynomialDecay(2E-4, 5000, 1E-4, .5)
        self.BATCH_SIZE = bs
        self.cycle_consistency_lambda = 1.
        self.padding_regular_lambda = .1
        self.padding_label_smoothing = .1
        self.identity_lambda = 1.
        self.vq_beta = 0
        self.D_Optimizer = tf.keras.optimizers.RMSprop(learning_rate=self.D_LEARNING_RATE, centered=True, global_clipnorm=1.)
        self.G_Optimizer = tf.keras.optimizers.RMSprop(learning_rate=self.G_LEARNING_RATE, centered=True, global_clipnorm=1.)

        self.D_Optimizer = tfa.optimizers.MovingAverage(self.D_Optimizer)
        self.G_Optimizer = tfa.optimizers.MovingAverage(self.G_Optimizer)
        
        _dataset_shuffling_buffer_size = 50000

        self.model_save_path = 'models'
        self.tb_writer = tf.summary.create_file_writer('tensorboard/')
        self.max_to_keep = 50
        self._saver_status = []
        
        # creating dataset iterator
        thermal = tf.data.Dataset.from_tensor_slices(tr_a).repeat().shuffle(_dataset_shuffling_buffer_size, reshuffle_each_iteration=True).batch(self.BATCH_SIZE, drop_remainder=True).map(lambda x: tf.one_hot(x, depth=21), num_parallel_calls=10).prefetch(5)
        non_thermal = tf.data.Dataset.from_tensor_slices(tr_b).repeat().shuffle(_dataset_shuffling_buffer_size, reshuffle_each_iteration=True).batch(self.BATCH_SIZE, drop_remainder=True).map(lambda x: tf.one_hot(x, depth=21), num_parallel_calls=10).prefetch(5)
        thermal4D = tf.data.Dataset.from_tensor_slices(tr_a).repeat().shuffle(_dataset_shuffling_buffer_size, reshuffle_each_iteration=True).batch(self.BATCH_SIZE, drop_remainder=True).map(lambda x: tf.one_hot(x, depth=21), num_parallel_calls=10).prefetch(5)
        non_thermal4D = tf.data.Dataset.from_tensor_slices(tr_b).repeat().shuffle(_dataset_shuffling_buffer_size, reshuffle_each_iteration=True).batch(self.BATCH_SIZE, drop_remainder=True).map(lambda x: tf.one_hot(x, depth=21), num_parallel_calls=10).prefetch(5)
        
        self.thermal_iter = iter(thermal)
        self.non_thermal_iter = iter(non_thermal)
        self.thermal4D_iter = iter(thermal)
        self.non_thermal4D_iter = iter(non_thermal)

        def thermal_data_generator():
            while(1):
                np.random.shuffle(_tr_a_ind)
                for i in _tr_a_ind:
                    yield tf.one_hot(self.therm_seq[i], depth=21)
                pass
            pass
        pass

        def nonthermal_data_generator():
            while(1):
                np.random.shuffle(_tr_b_ind)
                for i in _tr_b_ind:
                    yield tf.one_hot(self.nontherm_seq[i], depth=21)
                pass
            pass
        pass

    pass 

    def _seq_picker(self, seq_dataset, indx):
        return seq_dataset[indx]
    pass

    def _seq_sampler(self, seq_dataset, indx):
        return [seq_dataset[j] for j in indx]
    pass

    def _model_saver(self, iter_counter):
        self.therm2nontherm_G.save('{}/therm2nontherm_G.h5'.format(self.model_save_path), save_traces=True)
        self.nontherm2therm_G.save('{}/nontherm2therm_G.h5'.format(self.model_save_path), save_traces=True)
        self.therm_D.save('{}/therm_D.h5'.format(self.model_save_path), save_traces=True)
        self.nontherm_D.save('{}/nontherm_D.h5'.format(self.model_save_path), save_traces=True) 
        
        self._saver_status.append(iter_counter)
        self.therm2nontherm_G.save('{}/therm2nontherm_G_{}.h5'.format(self.model_save_path, iter_counter), save_traces=True)
        self.nontherm2therm_G.save('{}/nontherm2therm_G_{}.h5'.format(self.model_save_path, iter_counter), save_traces=True)
        self.therm_D.save('{}/therm_D_{}.h5'.format(self.model_save_path, iter_counter), save_traces=True)
        self.nontherm_D.save('{}/nontherm_D_{}.h5'.format(self.model_save_path, iter_counter), save_traces=True)

        if len(self._saver_status) == (self.max_to_keep + 1):
            self._saver_status.reverse()
            check = self._saver_status.pop()
            self._saver_status.reverse()
            for test_file in [
                              '{}/therm2nontherm_G_{}.h5'.format(self.model_save_path, check), \
                              '{}/nontherm2therm_G_{}.h5'.format(self.model_save_path, check), \
                              '{}/therm_D_{}.h5'.format(self.model_save_path, check), \
                              '{}/nontherm_D_{}.h5'.format(self.model_save_path, check)\
                             ]:
                if os.path.exists(test_file):
                    os.remove(test_file)   
                pass
            pass
        pass 
    pass 

    def _seq_iter_fecher(self):
        therm_data = next(self.thermal_iter)
        nontherm_data = next(self.non_thermal_iter)
        
        self.therm_data = therm_data
        self.nontherm_data = nontherm_data
    pass

    def triger(self):# oversampling for thermal dataset
            
        def G_loss():
            therm_data, nontherm_data = self.therm_data, self.nontherm_data 

            ##### the cycle gan will be considered as a specific form of VQVAE #####
            ### generating sequences and tranforming into index from
            therm2nontherm_logits = self.therm2nontherm_G(therm_data) 
            nontherm2therm_logits = self.nontherm2therm_G(nontherm_data)
            therm2nontherm_gens = tf.argmax(therm2nontherm_logits, axis=-1)                
            nontherm2therm_gens = tf.argmax(nontherm2therm_logits, axis=-1)
            therm2nontherm_onehot = tf.one_hot(therm2nontherm_gens, depth=21)
            nontherm2therm_onehot = tf.one_hot(nontherm2therm_gens, depth=21)

            ### cycle generating
            # zq = ze + tf.stop_gradient(zq - ze)
            therm2nontherm_q = tf.nn.softmax(therm2nontherm_logits) + tf.stop_gradient(therm2nontherm_onehot - tf.nn.softmax(therm2nontherm_logits))
            nontherm2therm_q = tf.nn.softmax(nontherm2therm_logits) + tf.stop_gradient(nontherm2therm_onehot - tf.nn.softmax(nontherm2therm_logits))
            cycle_therm2nontherm_logits = self.nontherm2therm_G(therm2nontherm_q)
            cycle_nontherm2therm_logits = self.therm2nontherm_G(nontherm2therm_q)
            cycle_therm2nontherm_gens = tf.argmax(cycle_therm2nontherm_logits, axis=-1)
            cycle_nontherm2therm_gens = tf.argmax(cycle_nontherm2therm_logits, axis=-1)
            cycle_therm2nontherm_onehot = tf.one_hot(cycle_therm2nontherm_gens, depth=21)
            cycle_nontherm2therm_onehot = tf.one_hot(cycle_nontherm2therm_gens, depth=21)
            cycle_therm2nontherm_q = tf.nn.softmax(cycle_therm2nontherm_logits) + tf.stop_gradient(cycle_therm2nontherm_onehot - tf.nn.softmax(cycle_therm2nontherm_logits))
            cycle_nontherm2therm_q = tf.nn.softmax(cycle_nontherm2therm_logits) + tf.stop_gradient(cycle_nontherm2therm_onehot - tf.nn.softmax(cycle_nontherm2therm_logits))

            ## padding regularizer
            cycle_therm2nontherm_padding = tf.slice(tf.nn.softmax(cycle_therm2nontherm_logits, axis=-1), [0, 0, 0], [-1, -1, 1])
            cycle_nontherm2therm_padding = tf.slice(tf.nn.softmax(cycle_nontherm2therm_logits, axis=-1), [0, 0, 0], [-1, -1, 1])
            therm_padding = tf.slice(therm_data, [0, 0, 0], [-1, -1, 1])
            nontherm_padding = tf.slice(nontherm_data, [0, 0, 0], [-1, -1, 1])
            therm_padding_g_in_seq_mask = 1 - therm_padding
            nontherm_padding_g_in_seq_mask = 1 - nontherm_padding
            padding_loss =  tf.math.reduce_sum(therm_padding * -tf.math.log(cycle_therm2nontherm_padding + 1E-12)) + \
                            tf.math.reduce_sum(nontherm_padding * -tf.math.log(cycle_nontherm2therm_padding + 1E-12)) + \
                            tf.math.reduce_mean(therm_padding_g_in_seq_mask * -tf.math.log(1 - cycle_therm2nontherm_padding + 1E-12)) + \
                            tf.math.reduce_mean(nontherm_padding_g_in_seq_mask * -tf.math.log(1 - cycle_nontherm2therm_padding + 1E-12)) 

            ### cycle loss
            cycle_consistency_loss_dec_t2n = tf.reduce_mean(tf.keras.losses.categorical_crossentropy(y_true=therm_data, y_pred=cycle_therm2nontherm_logits, from_logits=True, label_smoothing=.1) #* (.0 + (self.focal_alpha) * tf.pow(therm2nontherm_focal_reward, self.focal_pow))
                                                           )
            cycle_consistency_loss_dec_n2t = tf.reduce_mean(tf.keras.losses.categorical_crossentropy(y_true=nontherm_data, y_pred=cycle_nontherm2therm_logits, from_logits=True, label_smoothing=.1) #* (.0 + (self.focal_alpha) * tf.pow(nontherm2therm_focal_reward, self.focal_pow))
                                                           ) * 1
            cycle_consistency_loss_enc_t2n = tf.reduce_mean(tf.keras.losses.categorical_crossentropy(y_true=therm2nontherm_onehot, y_pred=tf.nn.softmax(therm2nontherm_logits, axis=-1), from_logits=False, label_smoothing=.1)) # * (.1 + self.focal_alpha) * tf.pow(therm2nontherm_focal_reward, self.focal_pow))
            cycle_consistency_loss_enc_n2t = tf.reduce_mean(tf.keras.losses.categorical_crossentropy(y_true=nontherm2therm_onehot, y_pred=tf.nn.softmax(nontherm2therm_logits, axis=-1), from_logits=False, label_smoothing=.1)) # * (.1 + self.focal_alpha) * tf.pow(nontherm2therm_focal_reward, self.focal_pow)) * 2
            
            
            cycle_consistency_loss_t2n = cycle_consistency_loss_dec_t2n + cycle_consistency_loss_enc_t2n * self.vq_beta
            cycle_consistency_loss_n2t = cycle_consistency_loss_dec_n2t + cycle_consistency_loss_enc_n2t * self.vq_beta

            ### identity generating
            identity_nontherm_logits = self.therm2nontherm_G(nontherm_data)
            identity_therm_logits    = self.nontherm2therm_G(therm_data)
            # focal reward for identify loss
            i_nontherm2nontherm_focal_reward = \
            1 - (tf.math.reduce_sum(tf.cast(tf.math.equal(tf.argmax(nontherm_data, axis=-1), tf.argmax(identity_nontherm_logits, axis=-1)), dtype=tf.float32) * tf.reshape(nontherm_padding_g_in_seq_mask, [-1, self.seq_len]), axis=-1, keepdims=True)/tf.reshape(tf.math.reduce_sum(nontherm_padding_g_in_seq_mask, axis=[-2,-1]), [-1,1])) #[-1, 1]
            i_therm2therm_focal_reward = \
            1 - (tf.math.reduce_sum(tf.cast(tf.math.equal(tf.argmax(therm_data, axis=-1), tf.argmax(identity_therm_logits, axis=-1)), dtype=tf.float32) * tf.reshape(therm_padding_g_in_seq_mask, [-1, self.seq_len]), axis=-1, keepdims=True)/tf.reshape(tf.math.reduce_sum(therm_padding_g_in_seq_mask, axis=[-2,-1]), [-1,1])) #[-1, 1]

            # padding regularization
            i_therm_padding = tf.slice(tf.nn.softmax(identity_therm_logits, axis=-1), [0, 0, 0], [-1, -1, 1])
            i_nontherm_padding = tf.slice(tf.nn.softmax(identity_nontherm_logits, axis=-1), [0, 0, 0], [-1, -1, 1])
            i_padding_loss = tf.math.reduce_sum(therm_padding * -tf.math.log(i_therm_padding + 1E-12)) + \
                             tf.math.reduce_sum(nontherm_padding * -tf.math.log(i_nontherm_padding + 1E-12)) + \
                             tf.math.reduce_sum(therm_padding_g_in_seq_mask * -tf.math.log(1 - i_therm_padding + 1E-12)) + \
                             tf.math.reduce_sum(nontherm_padding_g_in_seq_mask * -tf.math.log(1 - i_nontherm_padding + 1E-12))
            
            # global view
            identity_therm2nontherm_loss = tf.reduce_mean(
                                        tf.keras.losses.categorical_crossentropy(y_true=nontherm_data, y_pred=tf.nn.softmax(identity_nontherm_logits, axis=-1), from_logits=False,  label_smoothing=.1) #* (.0 + (self.focal_alpha) * tf.pow(i_nontherm2nontherm_focal_reward, self.focal_pow)) 
                                        ) * 1
            identity_nontherm2therm_loss = tf.reduce_mean(
                                        tf.keras.losses.categorical_crossentropy(y_true=therm_data, y_pred=tf.nn.softmax(identity_therm_logits, axis=-1), from_logits=False, label_smoothing=.1) #* (.0 + (self.focal_alpha) * tf.pow(i_therm2therm_focal_reward, self.focal_pow))
                                        ) 

            ### confusing D 
            therm_D_logits    = self.therm_D(nontherm2therm_q)
            nontherm_D_logits = self.nontherm_D(therm2nontherm_q)
            loss_D = tf.reduce_mean(tf.keras.losses.binary_crossentropy(y_pred=therm_D_logits, y_true=tf.ones(therm_D_logits.shape), from_logits=True, label_smoothing=.1)
                                   ) + \
                     tf.reduce_mean(tf.keras.losses.binary_crossentropy(y_pred=nontherm_D_logits, y_true=tf.ones(nontherm_D_logits.shape), from_logits=True, label_smoothing=.1)
                                   ) * 1
            
            total_loss = (loss_D + self.cycle_consistency_lambda * (cycle_consistency_loss_t2n + cycle_consistency_loss_n2t) + self.identity_lambda * (identity_therm2nontherm_loss + identity_nontherm2therm_loss) + self.padding_regular_lambda * (i_padding_loss + padding_loss))   
            
            self.c_G_loss = total_loss.numpy()
            with self.tb_writer.as_default():
                tf.summary.scalar('Generator loss', self.c_G_loss, step=self.iter_counter)
            pass

            return total_loss + tf.reduce_mean(self.therm2nontherm_G.losses) + tf.reduce_mean(self.nontherm2therm_G.losses)
        pass
                                
        def D_loss():
            ### fetch the sequence
            therm_data, nontherm_data = next(self.thermal4D_iter), next(self.non_thermal4D_iter)


            ### generating sequences and tranforming into index from
            therm2nontherm_logits = self.therm2nontherm_G(therm_data) 
            nontherm2therm_logits = self.nontherm2therm_G(nontherm_data)
            therm2nontherm_gens = tf.argmax(therm2nontherm_logits, axis=-1)
            nontherm2therm_gens = tf.argmax(nontherm2therm_logits, axis=-1)
            therm2nontherm_onehot = tf.one_hot(therm2nontherm_gens, depth=21)
            nontherm2therm_onehot = tf.one_hot(nontherm2therm_gens, depth=21)

            therm_r = self.therm_D(therm_data)
            nontherm_r = self.nontherm_D(nontherm_data)
            therm_f = self.therm_D(nontherm2therm_onehot)
            nontherm_f = self.nontherm_D(therm2nontherm_onehot)
            therm_in_nontherm_D = self.nontherm_D(therm_data)
            nontherm_in_therm_D = self.therm_D(nontherm_data)

            # 0 for fake samples and 1 for real samples
            # one-side label smoothing of .9
            total_loss = tf.reduce_mean(tf.keras.losses.binary_crossentropy(y_pred=therm_r, y_true=tf.ones(tf.shape(therm_r)), from_logits=True, label_smoothing=.1)) * 1. + \
                         tf.reduce_mean(tf.keras.losses.binary_crossentropy(y_pred=nontherm_r, y_true=tf.ones(tf.shape(nontherm_r)), from_logits=True, label_smoothing=.1)) * 1. + \
                         tf.reduce_mean(tf.keras.losses.binary_crossentropy(y_pred=therm_f, y_true=tf.zeros(tf.shape(therm_f)), from_logits=True, label_smoothing=.1)) + \
                         tf.reduce_mean(tf.keras.losses.binary_crossentropy(y_pred=nontherm_f, y_true=tf.zeros(tf.shape(nontherm_f)), from_logits=True, label_smoothing=.1)) + \
                         tf.reduce_mean(tf.keras.losses.binary_crossentropy(y_pred=therm_in_nontherm_D, y_true=tf.zeros(tf.shape(therm_in_nontherm_D)), from_logits=True, label_smoothing=.1)) + \
                         tf.reduce_mean(tf.keras.losses.binary_crossentropy(y_pred=nontherm_in_therm_D, y_true=tf.zeros(tf.shape(nontherm_in_therm_D)), from_logits=True, label_smoothing=.1))
            total_loss /= 6        

            self.c_D_loss = total_loss.numpy()
            with self.tb_writer.as_default():
                tf.summary.scalar('Discriminator loss', self.c_D_loss, step=self.iter_counter)
            pass
            
            return total_loss + tf.reduce_mean(self.therm_D.losses) + tf.reduce_mean(self.nontherm_D.losses)
        pass

        ### warm up
        self.D_Optimizer.learning_rate = 1E-9
        self.G_Optimizer.learning_rate = 1E-9
        for self.iter_counter in range(self.warmup_iteration_no):
            self._seq_iter_fecher()
            self.D_Optimizer.minimize(D_loss, self.therm_D.trainable_variables + self.nontherm_D.trainable_variables)
            self.G_Optimizer.minimize(G_loss, self.therm2nontherm_G.trainable_variables + self.nontherm2therm_G.trainable_variables)
        pass
        self.D_Optimizer.learning_rate = self.D_LEARNING_RATE
        self.G_Optimizer.learning_rate = self.G_LEARNING_RATE

        ### update 
        for self.iter_counter in range(self.iteration_no):
            self._seq_iter_fecher()

            if self.iter_counter % 1000 == 0:
                self.tb_writer.flush()
            pass
           
            if (self.iter_counter + 1) % 100 == 0:
                print('iter:{} loss_G:{} loss_D:{}'.format(self.iter_counter, self.c_G_loss, self.c_D_loss))
            
                ### save model
                self._model_saver(self.iter_counter)

                ### fetch the sequence
                therm_data, nontherm_data = self.therm_data, self.nontherm_data

                print('therm2nontherm:')
                print(sh.index2seq(tf.argmax(therm_data[0:1], axis=-1))[0])
                try:
                    print(sh.index2seq(tf.argmax(self.therm2nontherm_G(therm_data[0:1]), axis=-1))[0])
                except:
                    print("................")
                pass
                print('nontherm2therm:')
                print(sh.index2seq(tf.argmax(nontherm_data[0:1], axis=-1))[0])
                try:
                    print(sh.index2seq(tf.argmax(self.nontherm2therm_G(nontherm_data[0:1]), axis=-1))[0])
                except:
                    print(".................")
                pass
            pass

            if (self.iter_counter) % 1 == 0:
                self.D_Optimizer.minimize(D_loss, self.therm_D.trainable_variables + self.nontherm_D.trainable_variables)
            pass 
            self.G_Optimizer.minimize(G_loss, self.therm2nontherm_G.trainable_variables + self.nontherm2therm_G.trainable_variables)
        pass
    pass 
pass

