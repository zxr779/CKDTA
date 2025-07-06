# CKDTA
## Data

All data used in this paper are publicly available and can be accessed here:

• Davis and KIBA:[https://zenodo.org/records/10597707](https://zenodo.org/records/10597707)

• Filtered Davis: [https://github.com/cansyl/MDeePred](https://github.com/cansyl/MDeePred)

The construction of chemical element knowledge graph refers to [this reference](https://github.com/Fangyinfff/KCL).

## How to train

• Please download the data first, then unzip them, And create a data folder and put them in the data folder. • After deployment, please execute the following command to train the model. • Bash command:

```bash
bash training.sh
```

• Python command:

```python
python training_validation_Davis_KIBA.py
```

## File

• **create_data.py**: Load data from Filtered Davis、Davis and KIBA.   

• **training_validation_Davis_KIBA.py**: Train the model.   

• **predict_with_pretrained_model_Davis_KIBA.py**: Use the existing model files to test the data in the test set.  

• **environment.yml**: Include packages for the virtual environment dependencies   

• **utils.py**: Include functions for training, prediction, evaluation metrics, and more   

• **models**: Store the code for the model 

## Environment Setup

```bash
conda create -n CKDTA python=3.7
conda activate CKDTA
pip install -i https://pypi.tuna.tsinghua.edu.cn/simple rdkit
pip3 install torch==1.8.1+cu102 -f https://download.pytorch.org/whl/cu102/torch_stable.html
conda install cudatoolkit
pip install torch-scatter==2.0.8 -f https://pytorch-geometric.com/whl/torch-1.8.1%2Bcu102.html
pip install torch-sparse==0.6.12 -f https://pytorch-geometric.com/whl/torch-1.8.1%2Bcu102.html
pip install torch-cluster==1.5.9 -f https://pytorch-geometric.com/whl/torch-1.8.1%2Bcu102.html
pip install torch-spline-conv==1.2.1 -f https://pytorch-geometric.com/whl/torch-1.8.1%2Bcu102.html
pip install torch-geometric==2.1.0
pip install lifelines==0.27.3
pip install networkx==2.6.3
pip install transformers
```
