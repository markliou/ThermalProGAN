# ThermalProGAN
ThermalGen is an example application of Sequence-Based Unpaired-Sample of Novel Protein Inventor (SUNI) which is a CycleGan based architecture for generating the protein sequences with target functions.

This repository contains a Tensorflow implementation of SUNI algorithm which means the users can use the seqeunce dataset instead of the thermal-stable protein sequences used in ThermalGen for generating the proteins with the target funtions.  

## workflow
The source codes are located at the *ThermalGen_sourcecode* folder. The files are introduced below.
### Contents

*  **SUNI.py** : The architecture of SUNI. The training parameters and loss function of ThermalGen are recorded in this file. The neural networks are constructed by using Tensorflow mainly.
*  **sequence_handler.py** : The sequence manipulating module, including the one-hot handling, sequence handling (mainly based on Biopython).
*  **Inference.py** : This file give an example to inferencing protein by using ThermalGen. But the model file is too huge to provide by GitHub.
*  **main.py** : The main function to train ThermalGen
*  **Dockerfile** : The working enviroment for SUNI. The docker image can also be downloaded from DockerHub. The detail step will be given below.

### Requirements

#### Packages (refer from requirements.txt)
* biopython==1.79
* numpy==1.18.5
* tensorflow==2.6.0
* tensorflow-addons==0.14.0


#### Input data
* The protein sequences with FASTA representing. For example:

```
>tr|U3GMR0|U3GMR0_THEAI GyrB (Fragment) OS=Thermoplasma acidophilum OX=2303 GN=gyrB PE=4 SV=1
KLTGGLHGVGLHVVNALSKKLIAVVKHDGKIYYDIFEQGIPVSGLKTASDVSEIEKLGIKIQFPEHGAIIKFYPDPDIFE
TTEFSYETILARLTDLSYLNPQLTITFLDEASGRKDILHHEGGLIELVRHLSEGKEALMEPLYLKEEVDSHMVEFSLLYT
TDVQETLMSFVNNISTPEGGTHVAGFHQGLSRAIQDYARSNNKIK
>tr|U3GKK7|U3GKK7_THEAI GyrB (Fragment) OS=Thermoplasma acidophilum OX=2303 GN=gyrB PE=4 SV=1
FDKKVYKLTGGLHGVGLHVVNALSKKLIAVVKRDGKIYYDIFEQGIPVSGLKTASDVSEIEKLGIKIQFPEHGAIIKFYP
DPDIFETTEFSYETILARLTDLSYLNPQLTITFLDEASGRKDILHHEGGLIELVRHLSEGKEALMEP
>tr|Q67Q64|Q67Q64_SYMTH Transposase OS=Symbiobacterium thermophilum (strain T / IAM 14863) OX=292459 GN=STH1194 PE=4 SV=1
MELIAYLFGIRSDRRLVEEIRVNVAYRWFLGLGLTDPVPHFTTPGKNYSRRWKDSGLFEELFDHVVKQAIDAGYIDGRMI
FTDSSHLKANANKRRIAKEGTKGVTLEDIARARERHLAARRAEREECAANDDTEDGLLAAVNADREAHGLKPLPERKEDP
QPDLSVDEMTVSLTDPEAAMLRREGKPDGFHYLQHRTVDGRHGFILDVLVTSAAMTDAQVYPTCLSRVDRHGLKVEKVGV
DAGYNTLEVLHLLSKRGIQAAVAHRRHPSPKELMGKWRFKYDASRDAYRCPAKQWLTYVTTNRDGYRVYRSDASVCASCP
LLGQCTRSTTKQKVIHRHLYEHLREEAREFVKTDEGQRLAQRRRETVERSFADAKELHGLRYARYRGRKRVQHQCLVSAL
AQNLKKLALLESRRSSYALSA
```
ThermalGen is trained by normal proteins which come from the regular living  organisms and the thermal-stable proteins come from the thermalphilic living organisms.
## Usage
The file containes the sequences from one domain (e.g. normal proteins) and the files contains the sequences from another domain(e.g. thermal-stable proteins) are necessary. For training the customized files, the file names are needed to be specified in the *main.py*:
```python
def main():
    therm_training_file = 'thermophilic.fasta.uc100' # give the file name of one domain sequences
    nontherm_training_file = 'nonthermophilic.fasta.uc100.uc20'# give the file name of another domain sequences
```
After specifying the file names, the training process can be easily triggered. Before training, please make sure the training dataset is put with the main.py together:
```bash
mkdir models tensorboard
python main.py
```
The model will be automatically save every 100 steps and keep the save model number in 50. The models can be found at *models/* .
### using docker 
The training enviroments are provided in the dockerfile. If the docker image can be easily pulled from [DockerHub]([htt](https://hub.docker.com/layers/107021249/markliou/python3-tensorflow-gpu/thermalgen/images/sha256-228cfd8c6f9f4d24bbd3746c0b89f1c6636b46d4d4b67e44e82d3eb11bae3763?context=repo)) or built by your self:
```bash
# directly pull from DockerHub
docker pull markliou/python3-tensorflow-gpu:thermalgen
# build the image from dockerfile. You need to locate at the folder contain the dockerfile
docker build -t markliou/python3-tensorflow-gpu:thermalgen .
```
After you have the docker image, you can use the docker for training:
```bash
# suppose you are located at the folder of the thermalgen_sourcecode  
docker run -it -v `pwd`:/workspace python main.py
# this image can also used the GPUs. Suppose you have nvidia-docker2
docker run -it -v `pwd`:/workspace --gpus all python main.py
```
Since the nvidia-docker2 has different usgage, more detailed infomration can be found at https://github.com/NVIDIA/nvidia-docker

### Inferencing
To inference the sequence, you need to modify the code to specify the model location and the inferened sequence:
```python
def Inference(seq, model_path="./models/nontherm2therm_G.h5", seq_len = 512):
    ThermalGen = tf.keras.models.load_model("./models/nontherm2therm_G.h5") # specified the model here
    #print(ThermalGen.summary())
    
    candidate_seq = sh.seq2index(seq)
    candidate_seq = tf.keras.preprocessing.sequence.pad_sequences([candidate_seq], value=0, maxlen=seq_len)
    candidate_seq = tf.one_hot(candidate_seq, depth=21)

    Therm_seq = sh.index2seq(tf.argmax(ThermalGen(candidate_seq), axis=-1))[0]

    return Therm_seq
pass
```
In fact, the training process save models every 100 steps and will save the newest model as "therm2nontherm.h5" and "nontherm2therm.h5", so this modify would not be necessary if you do not modify the traning script too much. The next step, you need to give the sequence and inferece it:
```python
# give the sequence
hemoglobin_seq = 'MHSSIVLATVLFVAIASASKTRELCMKSLEHAKVGTSKEAKQDGIDLYKHMFEHYPAMKKYFKHRENYTPADVQKDPFFIKQGQNILLACHVLCATYDDRETFDAYVGELMARHERDHVKVPNDVWNHFWEHFIEFLGSKTTLDEPTKHAWQEIGKEFSHEISHHGRHSVRDHCMNSLEYIAIGDKEHQKQNGIDLYKHMFEHYPHMRKAFKGRENFTKEDVQKDAFFVNKDTRFCWPFVCCDSSYDDEPTFDYFVDALMDRHIKDDIHLPQEQWHEFWKLFAEYLNEKSHQHLTEAEKHAWSTIGEDFAHEADKHAKAEKDHHEGEHKEEHH'

# inference and pring the result 
Therm_seq = Inference(hemoglobin_seq)
print(Therm_seq)
```
Due to Github does not support large file storing currently, the nonthermal-to-thermal model is temporarily shared via [pCloud](https://u.pcloud.link/publink/show?code=XZYQg0VZknUTIn8gAY0bb9v7lsoOmVMC9IK7)
### the webservice
This part is working on. The website scripts are ready in the *web_service* folder, but there is still no suitable server to startup this website. If the website is ready, the part will be update immediately.