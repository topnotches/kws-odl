# KWS-ODL

## Credits
Credit where credit is due, thanks Cristian Cioflan for providing KWS-ON-PULP code, and Torben Hellreignnelaen for providing an x-vector project without requiring Kaldi :O

## Getting started

Run commands ./0* to set up data-dirs:

### ./00_get_data_speech_commands.sh
Downloads a dataset containing .wav files (dataset)

### ./01_setup_xvector_dirs.sh
Creates dir layout for x vector project (dataset_xvector)

### ./02_sc_to_vc_layout.py
Maps .wav from .wav dataset -> VoxCeleb format of xvector dataset (dataset_xvector/VoxCeleb).

### ./03_get_data_musan.sh
Gets and maps musan data into the xvector dataset dir (dataset_xvector/musan)

### ./04_get_data_rir.sh
Gets and maps rir data into the xvector dataset dir (dataset_xvector/rir_noises)

### ./05_gen_raw_mffcs.py
Converts the .wav dataset into raw MFCC bin files (processed in the same way as the done in the KWS-ON-PULP project by Cioflan)

### Setup X-Vector Conda Env
```
conda env create -f environment.yml
```

### xvector
At the top of xvector/main.py there are definitions to set which stages are to be run, in addition to other important parameters such as embeddings size (rank of LDA), after setting these, run:
```
cd xvector
python main.py
```

### kws training
For training the kws model run:
```
cd backbone
python main.py
```
### simulation
Currently only tests are supported for implemented layers, run:
```
cd simulation
make tests
```