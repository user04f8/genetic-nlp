# Summary

This repo contains the winning submission for the CS 506 kaggle competition, principally leveraging matrix factorization as a means of collaborative filtering with a fuzzy matching of n-grams leveraging the GloVe embedding.

Full details are available in `postmortem.pdf`.

Expirements and exploratory scripts used in the production of the final tools are available in `experiments/`. Move these to the repo root if you wish to reproduce them.

See below for reproduction of the winning submission.

# Setup

A full script to do all of the below is available in `./run_all.sh`, assuming you already have **Python 3.11.7** or a compatible version and have the `kaggle` CLI set up. I have also tested the core functionalities in a Python 3.12.7 conda environment; however, I recommend setting up Python 3.11.7. Note the below special instructions if you use conda in Conda setup.

- if this is not the commit in which this line was added, DELETE ALL cache/ and .joblib files to fully reproduce; if you're trying to skip to generate submission then run `python fuzzy_ngram_matrix_factorization.py` with the meta-hyperparameters (in the .yaml) to then generate the correct .joblibs
- Install/activate a virtual environment with Python 3.11.7
- Install `requirements.txt`
- Ensure the dataset is downloaded in `data/`, either as a *.zip or *.csv files
- Run the preprocessing script `preprocess_data.py`
- Download GloVe: `wget https://nlp.stanford.edu/data/wordvecs/glove.840B.300d.zip`
- (optional) reproduce the hyperparameter search via `python hyper_train.py` -- note that as of this commit, due to compute limitations the top hyperparameters has not been able to reproduce my intuition for what best parameters are
- Run the main train script via `python modular_train.py` (best hyperparameters) or `python fuzzy_ngram_matrix_factorization.py` (more fine manual control, also the file used in submission 1) 
- Generate the final csv file via `python fuzzy_ngram_generate_submission.py` (**Skip to this step if you just want to reproduce the final model test**)

The above is for the reproduction of the winning submission. If you'd like to reproduce the xgd boost submission, please see `run_all_submission_2.sh`.

## Conda setup

NOTES: if you are using conda and run into issues:
 - make sure you run the below installs, especially pytorch and pytorch-lightning, **before** doing pip install -r requirements.txt
 - make sure pip references and installs to your conda installation
 - try something like REQUESTS_CA_BUNDLE=/etc/ssl/certs/ca-certificates.crt if the below throws some SSL validation error

```
conda install pytorch pytorch-cuda=12.4 -c pytorch -c nvidia
conda install conda-forge::pytorch-lightning
conda install pandas numpy tiktoken gensim implicit joblib scikit-learn tqdm
pip install -r requirements.txt
```
