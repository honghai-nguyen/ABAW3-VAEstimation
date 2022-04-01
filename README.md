
:3rd_place_medal: [Affwild2-ABAW3 @ CVPRW 2022](https://ibug.doc.ic.ac.uk/resources/cvpr-2022-3rd-abaw/) -  Valence Arousal Estimation - PRL

### 1. Create environment
+ Create a python environment using conda or other tools.
+ Instead packages in requirements.txt with pip install -r requirements.txt, or manually.
### 2. Data preparation
+ Create 'testset' folder which contains test set release.
+ Create 'batch_1_2' folder which contains batch1 and batch2.
+ Run tools/data_preparation.py and tools/prepare_test_data.py
*Note: Change the path of the dataset.
### 3. Train k-fold
+ Edit 'root_project' and 'data_dir' in scripts/train_kfold.sh.
+ Run training k-fold:
```bash
bash  scripts/train_kfold.sh.
```
+ Edit 'data_dir' in tools/prepare_data_from_kfold.py and run it.
### 4. Traning GRU block with Local Attention
+ Edit 'root_project' and 'data_dir' in scripts/train_GRU_Att.sh.
+ Run training 
```bash
bash  scripts/train_GRU_Att.sh.
```

### 5. Traning GRU block combined with Transformer block.
+ Edit 'root_project' and 'data_dir' in scripts/train_GRU_Tran.sh.
+ Run training 
```bash
bash  scripts/train_GRU_Tran.sh.
```
