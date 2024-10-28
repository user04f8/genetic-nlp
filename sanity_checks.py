import pandas as pd

from preprocess_data import load_data

def sanity_check_submission(submission_file, test_df, num_examples=5):
    """
    Perform a sanity check on the submission by printing example data points from the test data
    and their corresponding predictions from the submission file.
    
    Parameters:
    - submission_file: Path to the submission.csv file.
    - test_df: The original test DataFrame (before processing).
    - num_examples: Number of examples to print.
    """
    # Load the submission file
    submission_df = pd.read_csv(submission_file)

    # Ensure we're not printing more examples than available in the submission
    num_examples = min(num_examples, len(submission_df))

    # Iterate over the submission and find matching entries in the test DataFrame
    for idx, (review_id, predicted_score) in enumerate(submission_df.values[:num_examples]):
        # Find the corresponding row in test_df by 'Id'
        matching_row = test_df.loc[test_df['Id'] == review_id]

        if matching_row.empty:
            print(f"Review ID {review_id} not found in test data.")
            continue

        # Extract relevant fields
        user_id = matching_row['UserId'].values[0]
        product_id = matching_row['ProductId'].values[0]
        text = matching_row['Text'].values[0]
        summary = matching_row['Summary'].values[0] if 'Summary' in matching_row else "N/A"

        # Display the example with predicted score
        print(f"Example {idx + 1}")
        print(f"Review ID: {review_id}")
        print(f"User ID: {user_id}")
        print(f"Product ID: {product_id}")
        print(f"Text: {text}")
        print(f"Summary: {summary}")
        print(f"Predicted Score: {predicted_score}")
        print('-' * 50)

if __name__ == '__main__':
    _, test_df = load_data()
    submission_file = 'submission.csv'
    sanity_check_submission(submission_file, test_df, num_examples=25)
