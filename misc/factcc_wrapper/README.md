## FactCC Wrapper

A factcc wrapper can be found under `abstractive-factual-tradeoff/misc/factcc_wrapper/`

### Download FactCC Code and Model Checkpoint

Download [factCC](https://github.com/salesforce/factCC) and put it in a directory, 
e.g., your home directory `${HOME}`:

```
cd ${HOME}
git clone https://github.com/salesforce/factCC
cd factCC
git checkout 6170a8c # for reproducibility 
```

Define `FACTCC_DIR` and add `factCC` to your `PYTHONPATH`:
```
export FACTCC_DIR=${HOME}/factCC/
export PYTHONPATH=${FACTCC_DIR}:${FACTCC_DIR}/modeling:$PYTHONPATH
```

Download and untar [checkpoint](https://storage.googleapis.com/sfr-factcc-data-research/factcc-checkpoint.tar.gz) and define FACTCC_MODEL:
```
cd ${FACTCC_DIR}
wget https://storage.googleapis.com/sfr-factcc-data-research/factcc-checkpoint.tar.gz
tar -xzf factcc-checkpoint.tar.gz
export FACTCC_MODEL=${PWD}/factcc-checkpoint
```

### Create a Conda environment and install dependencies

```
conda create --name factcc python=3.6
source activate factcc
pip install -r ~/abstractive-factual-tradeoff/misc/factcc_wrapper/requirements.txt
python -m spacy download en_core_web_lg
```

### Prepare data

Format your data into a json file named `input.jsonl` storing summary and article text per line:

```
{
  "article": "a dummy article",
  "summary": "a dummy summary"
}
```

### Run FactCC


```
command: sh run_factcc.sh input_file output_dir model_checkpoint gpu keys
- input_file: 
- output_dir: output directory 
- model_checkpoint: the directory storing the model checkpoint
- gpu: visible gpu ids, seperated by a comma
- keys: the key of article and summary text in the input file, seperated by a comma

example: sh run_factcc.sh input.jsonl output ${HOME}/factCC/checkpoint/ 0,1,2,3 article,summary
```
The factcc scores can be found at `output/factcc.csv`.

###  Notes
You may see warnings from using wandb. You may need to manually provide your choice of using wandb.

```
 wandb: WARNING W&B installed but not logged in.  Run `wandb login` or set the WANDB_API_KEY env variable.
 wandb: WARNING W&B installed but not logged in.  Run `wandb login` or set the WANDB_API_KEY env variable.
 wandb: WARNING W&B installed but not logged in.  Run `wandb login` or set the WANDB_API_KEY env variable.
 wandb: WARNING W&B installed but not logged in.  Run `wandb login` or set the WANDB_API_KEY env variable.
 wandb: WARNING W&B installed but not logged in.  Run `wandb login` or set the WANDB_API_KEY env variable.
 ...
 wandb: (1) Create a W&B account
 wandb: (2) Use an existing W&B account
 wandb: (3) Don't visualize my results
 wandb: Enter your choice: 3
 wandb: You chose "Don't visualize my results"
```
