import os
import pandas as pd

from .helper import *

from sentence_transformers import SentenceTransformer

import fitz

AVAILABLE_SECTIONS = ['011000',
                      '012100',
                      '012500',
                      '012900',
                      '013100',
                      '013300',
                      '014000',
                      '017419',
                      '017700',
                      '017900',
                      '000110',
                      '003100',
                      '011321',
                      '013200',
                      '015000',
                      '013113',
                      '013119',
                      '013216',
                      '013233',
                      '013543',
                      '013591',
                      '014523',
                      '015639',
                      '015716',
                      '016000',
                      '017300',
                      '019100']

KEYWORDS = {'security': ['security'], 'leed': ['leed'], 'shutdown': ['shut down', 'shut-down', 'shutdown'],
            'parking': ['parking'], 'permit': ['permit']}


def extract_keys_from_spec(spec, sections=None, score=0.7, six_digit_sections=False):
    if sections is None:
        sections = AVAILABLE_SECTIONS
    train_data = pd.read_parquet("train_data.parquet")

    train_data["section"] = train_data.section.apply(normalize_section)

    test_df = get_test_data_from_spec(spec)

    if not six_digit_sections:
        train_data["section"] = train_data["section"].apply(get_section_index)
        test_data_sections = test_df["section"]
        test_df["section"] = test_df["section"].apply(get_section_index)
        sections = [get_section_index(section) for section in sections]

    model = SentenceTransformer("all-MiniLM-L6-v2", cache_folder="/tmp/")

    section_prediction_df = parse_data(test_df, train_data, model, sections)

    prediction_df = pd.concat(list(section_prediction_df.values()))
    prediction_df = prediction_df[prediction_df.score > score]
    prediction_df["page num"] = prediction_df["page num"] + 1
    prediction_df = prediction_df.join(test_data_sections, how="left", lsuffix="_drop", rsuffix="")
    prediction_df.drop("section_drop", axis=1, inplace=True)

    return prediction_df


def extract_keys_from_spec_electrical_contractor(spec_title, spec, score=0.6):
    """
    Extracts key blocks from an electrical contractor's specification document.

    Parameters:
    spec_title (str): The title or filename of the specification document.
    score (float): The minimum score threshold for retaining the extracted keys (default is 0.6).

    Returns:
    pd.DataFrame: A DataFrame containing the filtered prediction scores, the highlights and the specification title.
    """
    # Paths to the training data CSV files for division 26
    final_train_data_path = 'services/training_data/final_train_data_division_26.csv'
    final_train_data = pd.read_csv(final_train_data_path)

    final_group_train_data_path = 'services/training_data/final_group_train_data_division_26.csv'
    final_group_train_data = pd.read_csv(final_group_train_data_path)

    delete_train_data_path = 'services/training_data/delete_train_data_division_26.csv'
    delete_train_data = pd.read_csv(delete_train_data_path)

    # Extract text blocks from the specification document specific to electrical contractors
    text_blocks = extract_division_26_blocks(spec)
    texts_to_classify = text_blocks

    # Load the pre-trained SentenceTransformer model
    model = SentenceTransformer("all-MiniLM-L6-v2", cache_folder="model/tmp/")

    # Parse the data and compute prediction scores for the extracted blocks
    test_scores = parse_data_electrical_contractor(texts_to_classify, final_train_data, final_group_train_data,
                                                   delete_train_data, model)

    if test_scores:
        # Convert the scores to a DataFrame
        prediction_df = pd.DataFrame(test_scores)
        # Filter out scores lower than the specified threshold
        prediction_df = prediction_df[prediction_df.score > score]
        # Add the specification title as a new column
        prediction_df['spec'] = os.path.basename(spec_title)
        return prediction_df


def extract_keys_from_spec_from_general_contractor(spec_title, spec, score=0.6):
    """
    Extracts key blocks from a specification document provided by a general contractor.

    Parameters:
    spec_title (str): The title or filename of the specification document.
    score (float): Threshold score for filtering results (default is 0.6).
    group (str): The group of sections to extract, default is 'security'.

    Returns:
    pd.DataFrame: A DataFrame containing the filtered prediction scores, the highlights and the specification title.
    """
    # Paths to the final and delete training data CSV files
    final_train_data_path = 'services/training_data/final_train_data_division_1.csv'
    final_train_data = pd.read_csv(final_train_data_path)
    delete_train_data_path = 'services/training_data/delete_train_data_division_1.csv'
    delete_train_data = pd.read_csv(delete_train_data_path)

    # Initialize an empty list to store text blocks
    text_blocks = []

    # Check if the group is 'security' and extract relevant sections
    for section in ['011400', '011419', '013528', '015200']:
        # Extract text blocks from the specified sections of the specification
        extracted_blocks = extract_division_1_blocks(spec, section, 'security')
        if extracted_blocks:
            text_blocks += extracted_blocks

    # Check if the group is 'leed' and extract relevant sections
    for section in ['015100']:
        # Extract text blocks from the specified sections of the specification
        extracted_blocks = extract_division_1_blocks(spec, section, 'leed')
        # print(extracted_blocks)
        if extracted_blocks:
            text_blocks += extracted_blocks

    # Check if the group is 'leed' and extract relevant sections
    for section in ['011100', '013528', '013546', '011419']:
        # Extract text blocks from the specified sections of the specification
        extracted_blocks = extract_division_1_blocks(spec, section, 'shutdown')
        if extracted_blocks:
            text_blocks += extracted_blocks

    # Check if the group is 'leed' and extract relevant sections
    for section in ['015200', '015526', '011419', '011100']:
        # Extract text blocks from the specified sections of the specification
        extracted_blocks = extract_division_1_blocks(spec, section, 'parking')
        if extracted_blocks:
            text_blocks += extracted_blocks

    # Check if the group is 'leed' and extract relevant sections
    for section in ['010146', '013100', '015526', '014100', '011813']:
        # Extract text blocks from the specified sections of the specification
        extracted_blocks = extract_division_1_blocks(spec, section, 'permit')
        if extracted_blocks:
            text_blocks += extracted_blocks
    # print(text_blocks)
    # Filter and prepare the text blocks for classification
    texts_to_classify = read_and_filter_blocks(text_blocks, KEYWORDS)
    # print(texts_to_classify)
    # Load the pre-trained SentenceTransformer model
    model = SentenceTransformer("all-MiniLM-L6-v2", cache_folder="model/tmp/")

    # Parse the data and compute prediction scores
    test_scores = parse_data_general_contractor(texts_to_classify, final_train_data, delete_train_data, model)

    if test_scores:
        # Convert the scores to a DataFrame
        prediction_df = pd.DataFrame(test_scores)

        # Filter rows where the prediction score is greater than the delete score
        prediction_df = prediction_df[prediction_df.score > prediction_df.delete_score]

        # Further filter rows to include only those with a delete score less than 0.5
        prediction_df = prediction_df[prediction_df.delete_score < 0.5]

        # Add the specification title as a new column
        prediction_df['spec'] = os.path.basename(spec_title)
        prediction_df_list = prediction_df.to_dict(orient='list')
        return prediction_df_list
