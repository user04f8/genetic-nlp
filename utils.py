import pandas as pd
from sklearn.model_selection import train_test_split

def preprocess_data(df):
    """
    Preprocesses the input dataframe by replacing None values in 'Summary' and 'Text' with an empty string.
    """
    # Replace None with an empty string in 'Summary' and 'Text' columns
    df['Summary'] = df['Summary'].fillna('')
    df['Text'] = df['Text'].fillna('')
    return df

def get_train_val_split(df, test_size=0.2, random_state=42):
    """
    Splits the dataframe into training and validation sets, ensuring that all 'ProductId' and 'UserId' in the
    validation set are also present in the training set.
    """
    print("Spitting the data...", end="")
    # Start with a basic random split of the dataset
    train_df, val_df = train_test_split(df, test_size=test_size, random_state=random_state)

    # Filter out rows in the validation set that have 'ProductId' or 'UserId' not present in the training set
    train_product_ids = set(train_df['ProductId'].unique())
    train_user_ids = set(train_df['UserId'].unique())

    valid_in_train = val_df['ProductId'].isin(train_product_ids) & val_df['UserId'].isin(train_user_ids)
    val_df = val_df[valid_in_train]

    # If validation set is reduced too much, re-sample from the training set
    while len(val_df) < len(df) * test_size:
        print(".", end="")
        additional_train_df, new_val_df = train_test_split(train_df, test_size=test_size, random_state=random_state)
        additional_valid_in_train = new_val_df['ProductId'].isin(train_product_ids) & new_val_df['UserId'].isin(train_user_ids)
        val_df = pd.concat([val_df, new_val_df[additional_valid_in_train]])
    print("")

    return train_df, val_df

# Example usage:
if __name__ == "__main__":
    from preprocess_data import load_data
    train_df, _ = load_data()

    # Preprocess the data
    train_df = preprocess_data(train_df)

    # Split the data
    train_df, val_df = get_train_val_split(train_df)
    
    print(f"Training set size: {len(train_df)}")
    print(f"Validation set size: {len(val_df)}")
