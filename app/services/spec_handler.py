import pandas as pd

from services.helper import normalize_section, get_section_index, get_test_data_from_spec, parse_data

from sentence_transformers import SentenceTransformer

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

def extract_keys_from_spec(spec, sections=AVAILABLE_SECTIONS, score=0.2, six_digit_sections=False):
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

    