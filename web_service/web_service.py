import tensorflow as tf
import tensorflow_addons as tfa
import numpy as np
import sequence_handler as sh

import flask
app = flask.Flask(__name__)

ThermalGen = tf.keras.models.load_model("./model/nontherm2therm_G.h5")
seq_len = 512
# print(ThermalGen.summary())

@app.route('/', methods=['GET',"POST"])
def index():
    seq = None
    if flask.request.method == 'POST':
        oseq = flask.request.values['seq']
        oseq = oseq.upper()

        candidate_seq = sh.seq2index(oseq.upper())
        candidate_seq = tf.keras.preprocessing.sequence.pad_sequences([candidate_seq], value=0, maxlen=seq_len)
        candidate_seq = tf.one_hot(candidate_seq, depth=21)

        seq = sh.index2seq(tf.argmax(ThermalGen(candidate_seq), axis=-1))[0]
        seq = seq.replace('-', '')
    pass

    if not seq:
        oseq = seq = "Sequence information is not ready"
    pass
    return flask.render_template('index.html', seq=seq, oseq=oseq)
pass

if __name__ == '__main__':
    app.debug = False
    app.run(host="0.0.0.0", port=8088)
pass
