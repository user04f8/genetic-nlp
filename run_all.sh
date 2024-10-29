# Assumes python already references version 3.11.7
python -m pip install -r requirements.txt
mkdir data
cd data
kaggle competitions download -c cs-506-midterm-fall-2024
cd ..
wget https://nlp.stanford.edu/data/wordvecs/glove.840B.300d.zip
mkdir cache
python preprocess_data.py
python fuzzy_ngram_matrix_factorization.py
python fuzzy_ngram_generate_submission.py
# diff submission.csv submission_1.csv
