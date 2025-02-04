DATA_FOLDER_PATH                        ='../dataset_xvector'
CHECKPOINT_PATH                         = './testlogs/lightning_logs/version_38/checkpoints/epoch=15-step=16000.ckpt'#'./testlogs/lightning_logs/version_0/checkpoints/epoch=14-step=14970.ckpt'
TRAIN_X_VECTOR_MODEL                    = False
EXTRACT_X_VECTORS                       = False
TRAIN_LDA                               = True
TEST_LDA                                = True
ALL_LDA                                 = True
EXPORT_ALL_LDA                          = True
USE_LDA                                 = True
TRAIN_PLDA                              = True
TEST_PLDA                               = True
X_VEC_EXTRACT_LAYER                     = 6
PLDA_RANK_F                             = 25
LDA_RANK_F                              = 64
EXTRACTED_XVECTOR_OUTPUT_PATH_TRAIN     = 'x_vectors/TRAINING_EXTRACTED.csv'
EXTRACTED_XVECTOR_OUTPUT_PATH_TEST      = 'x_vectors/TESTING_EXTRACTED.csv'
EXTRACTED_XVECTOR_OUTPUT_PATH_ALL       = 'x_vectors/ALL_EXTRACTED.csv'
EXTRACTED_LDA_OUTPUT_PATH               = 'embeddings'
EXTRACTED_LDA_OUTPUT_TRAIN_FILE_NAME    = 'embeddings_train_' + str(LDA_RANK_F)+'.csv'
EXTRACTED_LDA_OUTPUT_TEST_FILE_NAME     = 'embeddings_test_' + str(LDA_RANK_F)+'.csv'
EXTRACTED_LDA_OUTPUT_ALL_FILE_NAME      = 'embeddings_all_' + str(LDA_RANK_F)+'.csv'
EXTRACTED_LDA_OUTPUT_PATH_TRAIN         = EXTRACTED_LDA_OUTPUT_PATH+ "/" + EXTRACTED_LDA_OUTPUT_TRAIN_FILE_NAME
EXTRACTED_LDA_OUTPUT_PATH_TEST          = EXTRACTED_LDA_OUTPUT_PATH+ "/" + EXTRACTED_LDA_OUTPUT_TEST_FILE_NAME
EXTRACTED_LDA_OUTPUT_PATH_ALL           = EXTRACTED_LDA_OUTPUT_PATH+ "/" + EXTRACTED_LDA_OUTPUT_ALL_FILE_NAME
EXPORT_EMBEDDINGS_OUTPUT_DIR            = '../dataset_embeddings'
PLDA_MODEL_NAME                         = 'plda_model_' + str(LDA_RANK_F)


import os

import shutil
import numpy as np
import pandas as pd
import pytorch_lightning as pl
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.utils.tensorboard
import torchmetrics
from pytorch_lightning import loggers as pl_loggers
from pytorch_lightning.callbacks import ModelCheckpoint
from pytorch_lightning.callbacks.early_stopping import EarlyStopping
from torch.utils.data import DataLoader
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis
from sklearn.preprocessing import StandardScaler

import plda_classifier as pc
from config import Config
from dataset import Dataset
from plda_score_stat import plda_score_stat_object
from tdnn_layer import TdnnLayer

class XVectorModel(pl.LightningModule):
    def __init__(self, input_size=24,
                hidden_size=512,
                num_classes=2361,
                x_vector_size=512,
                x_vec_extract_layer=6,
                batch_size=256,
                learning_rate=0.001,
                batch_norm=True,
                dropout_p=0.0,
                augmentations_per_sample=2,
                data_folder_path='../dataset_xvector'):
        super().__init__()

        # Set up the TDNN structure including the time context of the TdnnLayer
        self.time_context_layers = nn.Sequential(
            TdnnLayer(input_size=input_size, output_size=hidden_size, context=[-2, -1, 0, 1, 2], batch_norm=batch_norm, dropout_p=dropout_p),
            TdnnLayer(input_size=hidden_size, output_size=hidden_size, context=[-2, 0, 2], batch_norm=batch_norm, dropout_p=dropout_p),
            TdnnLayer(input_size=hidden_size, output_size=hidden_size, context=[-3, 0, 3], batch_norm=batch_norm, dropout_p=dropout_p),
            TdnnLayer(input_size=hidden_size, output_size=hidden_size, batch_norm=batch_norm, dropout_p=dropout_p),
            TdnnLayer(input_size=hidden_size, output_size=1500, batch_norm=batch_norm, dropout_p=dropout_p)
        )
        self.segment_layer6 = nn.Linear(3000, x_vector_size)
        self.bn_fc1 = nn.BatchNorm1d(512, momentum=0.1, affine=False)
        self.segment_layer7 = nn.Linear(x_vector_size, x_vector_size)
        self.bn_fc2 = nn.BatchNorm1d(512, momentum=0.1, affine=False)
        self.output = nn.Linear(x_vector_size, num_classes)

        self.x_vec_extract_layer = x_vec_extract_layer
        self.batch_size = batch_size
        self.learning_rate = learning_rate

        self.dataset = Dataset(data_folder_path=data_folder_path, augmentations_per_sample=augmentations_per_sample)
        self.accuracy = torchmetrics.Accuracy()

        self.save_hyperparameters()

    # The statistic pooling layer
    def stat_pool(self, x):
        mean = torch.mean(x, 1)
        stand_dev = torch.std(x, 1)
        out = torch.cat((mean, stand_dev), 1)
        return out
        
    # The standard forward pass through the neural network
    def forward(self, x):
        out = self.time_context_layers(x)

        out = self.stat_pool(out)

        out = F.relu(self.bn_fc1(self.segment_layer6(out)))
        out = F.relu(self.bn_fc2(self.segment_layer7(out)))
        
        out = self.output(out)
        return out

    # This method is used to generate the x-vectors for the LDA/PLDA
    # It is the same as the usual forward method exept it stops passing the
    # input through the layers at the specified x_vec_extract_layer
    # Finally it returns the x-vectors instead of the usual output
    def extract_x_vec(self, x):
        out = self.time_context_layers.forward(x)

        out = self.stat_pool(out)

        if(self.x_vec_extract_layer == 6):
            x_vec = self.bn_fc1(self.segment_layer6.forward(out))
        elif(self.x_vec_extract_layer == 7):
            out = self.bn_fc1(F.relu(self.segment_layer6.forward(out)))
            x_vec = self.bn_fc2(self.segment_layer7.forward(out))
        else:
            x_vec = self.bn_fc1(self.segment_layer6.forward(out))
            
            
        return x_vec

    # Train the model
    def training_step(self, batch, batch_index):
        samples, labels, id = batch
        outputs = self(samples.float())
        loss = F.cross_entropy(outputs, labels)
        return {'loss': loss, 'train_preds': outputs, 'train_labels': labels, 'train_id': id}

    # Log training loss and accuracy with the logger
    def training_step_end(self, outputs):
        self.log('train_step_loss', outputs['loss'])
        accuracy = self.accuracy(outputs['train_preds'], outputs['train_labels'])
        self.log('train_step_acc', self.accuracy)
        return {'loss': outputs['loss'], 'acc': accuracy}

    def training_epoch_end(self, outputs):
        if self.current_epoch == 0:  # Log the graph only on the first epoch
            # Ensure the sample has the correct dimensions (batch_size, seq_len, input_size)
            sample = torch.rand((1, 299, 24), device=self.device)
            self.logger.experiment.add_graph(self, sample)

        # Log histograms for parameters
        for name, params in self.named_parameters():
            self.logger.experiment.add_histogram(name, params, self.current_epoch)

    # Calculate loss of validation data to check if overfitting
    def validation_step(self, batch, batch_index):
        samples, labels, id = batch
        outputs = self(samples.float())
        loss = F.cross_entropy(outputs, labels)
        return {'loss': loss, 'val_preds': outputs, 'val_labels': labels, 'val_id': id}

    # Log validation loss and accuracy with the logger
    def validation_step_end(self, outputs):
        self.log('val_step_loss', outputs['loss'])
        accuracy = self.accuracy(outputs['val_preds'], outputs['val_labels'])
        self.log('val_step_acc', self.accuracy)
        return {'loss': outputs['loss'], 'acc': accuracy}
    
    # The test step here is NOT used as a test step!
    # Instead it is used to extract the x-vectors
    def test_step(self, batch, batch_index):
        samples, labels, id = batch
        x_vecs = self.extract_x_vec(samples.float())
        return [(x_vecs, labels, id)]

    # After all x-vectros are generated append them to the predefined list
    def test_epoch_end(self, test_step_outputs):
        for batch_output in test_step_outputs:
            for x_vec, label, id in batch_output:
                for x, l, i in zip(x_vec, label, id):
                    x_vector.append((i, int(l.cpu().numpy()), np.array(x.cpu().numpy(), dtype=np.float64)))
        return test_step_outputs
    
    def configure_optimizers(self):
        return torch.optim.Adam(self.parameters(), lr=self.learning_rate)

    # Load only the training data
    def train_dataloader(self):
        self.dataset.load_data(train=True)
        train_data_loader = DataLoader(dataset=self.dataset, batch_size=self.batch_size, num_workers=4, shuffle=True)
        return train_data_loader

    # Load only the validation data
    def val_dataloader(self):
        self.dataset.load_data(val=True)
        val_data_loader = DataLoader(dataset=self.dataset, batch_size=self.batch_size, num_workers=4, shuffle=False)
        return val_data_loader

    # Load either both training and validation or test data for extracting the x-vectors
    # In 'train' mode extract x-vectors for PLDA training, in 'test' mode for testing PLDA
    def test_dataloader(self):
        if(extract_mode == 'train'):
            self.dataset.load_data(train=True, val=True)
            test_data_loader = DataLoader(dataset=self.dataset, batch_size=self.batch_size, num_workers=4, shuffle=False)
        if(extract_mode == 'test'):
            self.dataset.load_data(test=True)
            test_data_loader = DataLoader(dataset=self.dataset, batch_size=self.batch_size, num_workers=4, shuffle=False)
        if(extract_mode == 'all'):
            self.dataset.load_data(all=True)
            test_data_loader = DataLoader(dataset=self.dataset, batch_size=self.batch_size, num_workers=4, shuffle=False)
        return test_data_loader



if __name__ == "__main__":

    # Adjust parameters of model, PLDA, training etc. here
    # Set your own data folder path here!
    # VoxCeleb MUSAN and RIR must be in the same data/ directory!
    # It is also possible to execute only select parts of the program by adjusting:
    # train_x_vector_model, extract_x_vectors, train_plda and test_plda
    # When running only later parts of the program a checkpoint_path MUST be given and
    # earlier parts of the programm must have been executed at least once
    print('setting up model and trainer parameters')
    config = Config(data_folder_path = DATA_FOLDER_PATH,
                    checkpoint_path = CHECKPOINT_PATH,
                    train_x_vector_model = TRAIN_X_VECTOR_MODEL,
                    extract_x_vectors = EXTRACT_X_VECTORS,
                    train_plda = TRAIN_PLDA,
                    test_plda = TEST_PLDA,
                    x_vec_extract_layer = X_VEC_EXTRACT_LAYER,
                    plda_rank_f = PLDA_RANK_F,
                    lda_rank_f = LDA_RANK_F,
                    train_lda = TRAIN_LDA,
                    test_lda = TEST_LDA,
                    all_lda = ALL_LDA,
                    export_all_lda = EXPORT_ALL_LDA
                    )

    # Define model and trainer
    tb_logger = pl_loggers.TensorBoardLogger(save_dir="testlogs/")
    early_stopping_callback = EarlyStopping(monitor="val_step_loss", mode="min")
    checkpoint_callback = ModelCheckpoint(monitor='val_step_loss', save_top_k=10, save_last=True, verbose=True)

    if(config.checkpoint_path == 'none'):
        model = XVectorModel(input_size=config.input_size,
                            hidden_size=config.hidden_size,
                            num_classes=config.num_classes,
                            x_vector_size=config.x_vector_size,
                            x_vec_extract_layer=config.x_vec_extract_layer,
                            batch_size=config.batch_size,
                            learning_rate=config.learning_rate,
                            batch_norm=config.batch_norm,
                            dropout_p=config.dropout_p,
                            augmentations_per_sample=config.augmentations_per_sample,
                            data_folder_path=config.data_folder_path)
    else:
        model = XVectorModel.load_from_checkpoint(config.checkpoint_path)
    model.dataset.init_samples_and_labels()

    trainer = pl.Trainer(callbacks=[early_stopping_callback, checkpoint_callback],
                        logger=tb_logger,
                        log_every_n_steps=1,
                        #accelerator='cpu',#TODO delete
                        accelerator='gpu', devices=[0],
                        max_epochs=config.num_epochs)
                        #small test adjust options: fast_dev_run=True, limit_train_batches=0.0001, limit_val_batches=0.001, limit_test_batches=0.002



    # Train the x-vector model
    if(config.train_x_vector_model):
        print('training x-vector model')
        if(config.checkpoint_path == 'none'):
            trainer.fit(model)
        else:
            trainer.fit(model, ckpt_path=config.checkpoint_path)
            
        print("Done with training x-vector extraction model!")
        print()
        print()


   
    # Extract the x-vectors
    if(config.extract_x_vectors):
        print('extracting x-vectors')
        if not os.path.exists('x_vectors'):
            os.makedirs('x_vectors')
        # Extract the x-vectors for training the PLDA classifier and save to csv
        
        x_vector = []
        extract_mode = 'train'
        if(config.train_x_vector_model):
            trainer.test(model)
            x_vector = pd.DataFrame(x_vector)
            x_vector.to_csv(EXTRACTED_XVECTOR_OUTPUT_PATH_TRAIN)
        elif(config.checkpoint_path != 'none'):
            trainer.test(model, ckpt_path=config.checkpoint_path)
            x_vector = pd.DataFrame(x_vector)
            x_vector.to_csv(EXTRACTED_XVECTOR_OUTPUT_PATH_TRAIN)
        else:
            print('could not extract train x-vectors')

        # Extract the x-vectors for testing the PLDA classifier and save to csv
        x_vector = []
        extract_mode = 'test'
        if(config.train_x_vector_model):
            trainer.test(model)
            x_vector = pd.DataFrame(x_vector)
            x_vector.to_csv(EXTRACTED_XVECTOR_OUTPUT_PATH_TEST)
        elif(config.checkpoint_path != 'none'):
            trainer.test(model, ckpt_path=config.checkpoint_path)
            x_vector = pd.DataFrame(x_vector)
            x_vector.to_csv(EXTRACTED_XVECTOR_OUTPUT_PATH_TEST)
        else:
            print('could not extract test x-vectors')
        
        x_vector = []
        extract_mode = 'all'
        if(config.train_x_vector_model):
            trainer.test(model)
            x_vector = pd.DataFrame(x_vector)
            x_vector.to_csv(EXTRACTED_XVECTOR_OUTPUT_PATH_ALL)
        elif(config.checkpoint_path != 'none'):
            trainer.test(model, ckpt_path=config.checkpoint_path)
            x_vector = pd.DataFrame(x_vector)
            x_vector.to_csv(EXTRACTED_XVECTOR_OUTPUT_PATH_ALL)
        else:
            print('could not extract all x-vectors')
        print()
        print()
    

    if config.train_lda:
        print('Applying LDA to training x-vectors...')
        if not os.path.exists(EXTRACTED_LDA_OUTPUT_PATH):
            os.makedirs(EXTRACTED_LDA_OUTPUT_PATH)
        
        x_vectors_train = pd.read_csv(EXTRACTED_XVECTOR_OUTPUT_PATH_TRAIN)
        x_vectors_train.columns = ['index', 'id', 'label', 'xvector']
        x_id_train = np.array(x_vectors_train['id'])
        x_label_train = np.array(x_vectors_train['label'], dtype=int)
        x_vec_train = np.array([np.array(x_vec[1:-1].split(), dtype=np.float64) for x_vec in x_vectors_train['xvector']])
        
        # Apply LDA
        lda = LinearDiscriminantAnalysis(n_components=config.lda_rank_f)
        x_vec_train_lda = lda.fit_transform(x_vec_train, x_label_train)
        
        # Standardize LDA-transformed x-vectors
        scaler = StandardScaler()
        x_vec_train_lda_std = scaler.fit_transform(x_vec_train_lda)
        
        # Save standardized LDA-transformed x-vectors
        x_vectors_train['xvector'] = [str(list(x)) for x in x_vec_train_lda_std]
        x_vectors_train.to_csv(EXTRACTED_LDA_OUTPUT_PATH_TRAIN, index=False)
        print("Done with training LDA and standardization!")
        print()
        print()

    if config.test_lda:
        print('Applying LDA to test x-vectors...')
        if not os.path.exists(EXTRACTED_LDA_OUTPUT_PATH):
            os.makedirs(EXTRACTED_LDA_OUTPUT_PATH)
        
        x_vectors_test = pd.read_csv(EXTRACTED_XVECTOR_OUTPUT_PATH_TEST)
        x_vectors_test.columns = ['index', 'id', 'label', 'xvector']
        x_id_test = np.array(x_vectors_test['id'])
        x_label_test = np.array(x_vectors_test['label'], dtype=int)
        x_vec_test = np.array([np.array(x_vec[1:-1].split(), dtype=np.float64) for x_vec in x_vectors_test['xvector']])
        
        # Apply LDA
        x_vec_test_lda = lda.transform(x_vec_test)
        
        # Standardize LDA-transformed x-vectors
        x_vec_test_lda_std = scaler.transform(x_vec_test_lda)
        
        # Save standardized LDA-transformed x-vectors
        x_vectors_test['xvector'] = [str(list(x)) for x in x_vec_test_lda_std]
        x_vectors_test.to_csv(EXTRACTED_LDA_OUTPUT_PATH_TEST, index=False)
        print("Done parsing test-set with LDA and standardization!")
        print()
        print()

    if config.all_lda:
        print('Applying LDA to ALL x-vectors...')
        if not os.path.exists(EXTRACTED_LDA_OUTPUT_PATH):
            os.makedirs(EXTRACTED_LDA_OUTPUT_PATH)
        
        x_vectors_all = pd.read_csv(EXTRACTED_XVECTOR_OUTPUT_PATH_ALL)
        x_vectors_all.columns = ['index', 'id', 'label', 'xvector']
        x_id_all = np.array(x_vectors_all['id'])
        x_label_all = np.array(x_vectors_all['label'], dtype=int)
        x_vec_all = np.array([np.array(x_vec[1:-1].split(), dtype=np.float64) for x_vec in x_vectors_all['xvector']])
        
        # Apply LDA
        x_vec_all_lda = lda.transform(x_vec_all)
        
        # Standardize LDA-transformed x-vectors
        x_vec_all_lda_std = scaler.transform(x_vec_all_lda)
        
        # Save standardized LDA-transformed x-vectors
        x_vectors_all['xvector'] = [str(list(x)) for x in x_vec_all_lda_std]
        x_vectors_all.to_csv(EXTRACTED_LDA_OUTPUT_PATH_ALL, index=False)
        print("Done parsing all x-vectors with LDA and standardization!")
        print()
        print()

    if(config.export_all_lda): 
        
        if not os.path.exists(EXPORT_EMBEDDINGS_OUTPUT_DIR):
            os.makedirs(EXPORT_EMBEDDINGS_OUTPUT_DIR)
        else:
            shutil.rmtree(EXPORT_EMBEDDINGS_OUTPUT_DIR)
            os.makedirs(EXPORT_EMBEDDINGS_OUTPUT_DIR)
        gsc_words = [name for name in os.listdir("../dataset")]
        
        for wordname in gsc_words:
            if (wordname[0] != "_" and os.path.isdir("../dataset/" + wordname)):
                os.makedirs(EXPORT_EMBEDDINGS_OUTPUT_DIR + '/' + wordname)
                
        
        print('EXPORT EMBEDDINGS TO GSC FORMAT: loading lda reduced x_vector data with dim: ' + str(config.lda_rank_f))
        print("Loading data from: " + str(EXTRACTED_LDA_OUTPUT_PATH_ALL))
        x_vectors_all = pd.read_csv(EXTRACTED_LDA_OUTPUT_PATH_ALL)
        
        for index, row in x_vectors_all.iterrows():
            id = row.get('id')
            embeddings = row.get('xvector')
            xvector_filename = id.split('/')[-1].split('.')[0]
            word = xvector_filename.split('_')[-1]
            gsc_filename = ""
            for splitword in xvector_filename.split('_')[:-1]:
                gsc_filename = gsc_filename + splitword + '_'
            gsc_filename = gsc_filename[:-1]
            embeddings_list = eval(embeddings)  # Be cautious with eval if input is not controlled
            embeddings_df = pd.DataFrame([embeddings_list])  # Convert to DataFrame
            embeddings_df.to_csv(EXPORT_EMBEDDINGS_OUTPUT_DIR + '/' + word + '/' +gsc_filename+'.csv', index=False)
            

    if(config.train_plda):
        if not os.path.exists('plda'):
            os.makedirs('plda')
        # Extract the x-vectors, labels and id from the csv
        if (USE_LDA):
            print('PLDA TRAIN: loading lda reduced x_vector data with dim: ' + str(config.lda_rank_f))
            print("Loading data from: " + str(EXTRACTED_LDA_OUTPUT_PATH_TRAIN))
            x_vectors_train = pd.read_csv(EXTRACTED_LDA_OUTPUT_PATH_TRAIN)
        else:
            print('PLDA TRAIN: loading x_vector data with plda dim: ' + str(config.lda_rank_f))
            print("Loading data from: " + str(EXTRACTED_XVECTOR_OUTPUT_PATH_TRAIN))
            x_vectors_train = pd.read_csv(EXTRACTED_XVECTOR_OUTPUT_PATH_TRAIN)
            x_vectors_train.columns = ['index', 'id', 'label', 'xvector']
            

        x_id_train = np.array(x_vectors_train['id'])
        x_label_train = np.array(x_vectors_train['label'], dtype=int)
        
        x_vec_train = np.array([np.array(x_vec.replace(",", "")[1:-1].split(), dtype=np.float64) for x_vec in x_vectors_train['xvector']])

        # Generate x_vec stat objects
        print('generating x_vec stat objects')
        # print("x_vec_train: ")
        # print(x_vec_train)
        # print("x_label_train: ")
        # print(x_label_train)
        # print("x_id_train: ")
        # print(x_id_train)
        tr_stat = pc.get_train_x_vec(x_vec_train, x_label_train, x_id_train)

        # Train plda
        print('training plda')
        plda = pc.setup_plda(rank_f=config.lda_rank_f, nb_iter=10)
        plda = pc.train_plda(plda, tr_stat)
        pc.save_plda(plda, PLDA_MODEL_NAME)
        print("Done with training PLDA!")
        print()
        print()
        



    if(config.test_plda):
        # Extract the x-vectors, labels and id from the csv
        if (USE_LDA):
            print('PLDA TEST: loading lda reduced x_vector data with dim: ' + str(config.lda_rank_f))
            print("Loading data from: " + str(EXTRACTED_LDA_OUTPUT_PATH_TEST))
            x_vectors_test = pd.read_csv(EXTRACTED_LDA_OUTPUT_PATH_TEST)
        else:
            print('PLDA TEST: loading x_vector data with plda dim: ' + str(config.lda_rank_f))
            print("Loading data from: " + str(EXTRACTED_XVECTOR_OUTPUT_PATH_TEST))
            x_vectors_test = pd.read_csv(EXTRACTED_XVECTOR_OUTPUT_PATH_TEST)
            x_vectors_test.columns = ['index', 'id', 'label', 'xvector']
        score = plda_score_stat_object(x_vectors_test)

        # Test plda
        print('testing plda')
        if(not config.train_plda):
            plda = pc.load_plda('plda/'+PLDA_MODEL_NAME+'.pickle')
        print(config.data_folder_path + '/VoxCeleb/veri_test2.txt')
        score.test_plda(plda, config.data_folder_path + '/VoxCeleb/veri_test2.txt')

        # Calculate EER and minDCF
        print('calculating EER and minDCF')
        score.calc_eer_mindcf()
        print('EER: ', score.eer, '   threshold: ', score.eer_th)
        print('minDCF: ', score.min_dcf, '   threshold: ', score.min_dcf_th)

        # Generate images for tensorboard
        #score.plot_images(tb_logger.experiment)

        pc.save_plda(score, 'plda_score') # FIXME TODO: better naming if needed
        print("Done with testing PLDA!")
        print()
        print()


    print('DONE')
'''
Notes: TODO remove

screen commands reminder:
-------------------------
screen          start screen
screen -list    list screens
ctrl+a d        detach from current screen
screen -r       reatach to screen
ctrl+a c        create new window
ctrl+a "        show windows
exit            exit/kill window
ctrl+a A        rename window
ctrl+a H        create log file/toggle logging

my data used
153516 sample each 3 sec
460548 sec
7676 min
127 h

total data available
153516 sample average 8.4 sec
1265760 sec
21096 min
350 h
'''
