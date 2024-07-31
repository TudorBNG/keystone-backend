import re
import pandas as pd
from sentence_transformers import util
import numpy as np


section_regex = [
    # "[0-9][0-9][0-9][0-9]00"
    # "[0-9][0-9][0-9][0-9]00",
    # "[0-9][0-9] [0-9][0-9]00",
    # "[0-9][0-9] [0-9][0-9] 00",
    "[0-9]{6}\ \-\ [0-9]{1}",
    "[0-9][0-9]\ [0-9][0-9]\ [0-9][0-9]\ \-\ [0-9]{1}",
]

# remove the ending of the section, replace spaces and add 2 zeroes at the end
def trim_section(section: str):
    return re.sub(r'\-[0-9]{1}', '', section.replace(' ', ''))[:4] + '00'


# format the sections, remove duplicates and sort them
def sections_parser(sections):
    trimmed_sections = list(map(trim_section, sections))
    unique_sections = list(dict.fromkeys(trimmed_sections))
    unique_sections.sort()

    return unique_sections


# in the PDF's text, find all sections that match the pattern
def find_sections_in_page(text):
    temp_list = []
    for pattern in section_regex:
        discovered_match = re.findall(pattern, text)
        
        if discovered_match:
            temp_list += discovered_match

    return temp_list

# extract the sections --- DEPRECATED
def map_sections_to_page(sliced_text, page_num, section_candidates):
    for block in sliced_text: 
        line = block[4]
        for pattern in section_regex: 
            discovered_match = re.findall(pattern, line)
            if discovered_match:
                if hasattr(section_candidates, str(page_num)):
                    section_candidates[page_num] += discovered_match
                else:
                    section_candidates[page_num] = [discovered_match]
    return section_candidates

def normalize_section(section):
    return section.replace(" ", "")

def get_section_index(section):
    return section[:4]

def remove_new_line(block):
    return block[4].replace("\n", "")

def get_quad(block):
    return block[0:4]

def parse_text_blocks(text_blocks):
    contents = []
    quads = []
    
    for block in text_blocks:
        contents.append(remove_new_line(block))
        quads.append(get_quad(block))

    return contents, quads


def get_test_data_from_spec(spec):
    test_data = []

    page_section_map = {i: [] for i in range(len(spec))}

    for index, page in enumerate(spec):

        text = page.get_text("blocks", sort=True)
        contents, quads = parse_text_blocks(text)

        page_section_map = map_sections_to_page(text, index, page_section_map)

        if len(page_section_map[index]):
            section = page_section_map[index][0]

            if len(contents):
                for content, quad in zip(contents, quads):
                    test_data.append({"page num": index, "section": section, "text": content, "quads": quad})
    
    test_df = pd.DataFrame(test_data)
    test_df["section"] = test_df.section.apply(normalize_section)

    return test_df


def parse_data(test_df, train_data, model, sections):
    section_prediction_df = {}

    for index, section in enumerate(sections):
        train_data_temp = train_data[train_data.section==section].copy()
        train_data_sentences = train_data_temp["highlight"].tolist()
        train_model_embed = model.encode(train_data_sentences, convert_to_tensor=True)

        predict_content = test_df[test_df.section==section].copy()

        predict_sentence = predict_content["text"].tolist()

        if len(predict_sentence):
            predict_model_embed = model.encode(predict_sentence, convert_to_tensor=True)
            scores = util.cos_sim(predict_model_embed, train_model_embed)

            predict_content["score"] = np.array(scores.cpu()).max(axis=1)

            section_prediction_df[section] = predict_content

    return section_prediction_df
