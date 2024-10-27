# Assumes python already references version 3.11.7
python -m pip install -r requirements.txt
cd data
kaggle competitions download -c cs-506-midterm-fall-2024
cd ..
python preprocess_data.py
python fuzzy_ngram_matrix_factorization.py