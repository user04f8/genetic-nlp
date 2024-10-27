# Setup

A full script to do all of the below is available in `./run_all.sh`, assuming you already have **Python 3.11.7** or a compatible version and have the `kaggle` CLI set up.

- Install/activate a virtual environment with Python 3.11.7
- Install `requirements.txt`
- Ensure the dataset is downloaded in `data/`, either as a *.zip or *.csv files
- Run the preprocessing script `preprocess_data.py`
- Download GloVe: `wget https://nlp.stanford.edu/data/wordvecs/glove.840B.300d.zip`
- Run the main train script via `python fuzzy_ngram_matrix_factorization.py`
- Generate the final csv file via `python fuzzy_ngram_generate_submission.py` (**Skip to this step if you just want to reproduce the final model test**)
