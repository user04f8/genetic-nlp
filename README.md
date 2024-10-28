# Setup

A full script to do all of the below is available in `./run_all.sh`, assuming you already have **Python 3.11.7** or a compatible version and have the `kaggle` CLI set up.

- if this is not the commit in which this line was added, DELETE ALL cache/ and .joblib files to fully reproduce; if you're trying to skip to generate submission then run `python fuzzy_ngram_matrix_factorization.py` with the meta-hyperparameters (in the .yaml) to then generate the correct .joblibs
- Install/activate a virtual environment with Python 3.11.7
- Install `requirements.txt`
- Ensure the dataset is downloaded in `data/`, either as a *.zip or *.csv files
- Run the preprocessing script `preprocess_data.py`
- Download GloVe: `wget https://nlp.stanford.edu/data/wordvecs/glove.840B.300d.zip`
- (optional) reproduce the hyperparameter search via `python hyper_train.py` -- note that as of this commit, due to compute limitations the top hyperparameters has not been able to reproduce my intuition for what best parameters are
- Run the main train script via `python modular_train.py` (best hyperparameters) or `python fuzzy_ngram_matrix_factorization.py` (more fine manual control, also the file used in submission 1) 
- Generate the final csv file via `python fuzzy_ngram_generate_submission.py` (**Skip to this step if you just want to reproduce the final model test**)
