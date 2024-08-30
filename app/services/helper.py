import re
import pandas as pd
from sentence_transformers import util
import numpy as np
import fitz
import string

section_extraction_regex = [
    # "[0-9][0-9][0-9][0-9]00"
    # "[0-9][0-9][0-9][0-9]00",
    # "[0-9][0-9] [0-9][0-9]00",
    # "[0-9][0-9] [0-9][0-9] 00",
    "[0-9]{6}\ \-\ [0-9]{1}",
    "[0-9][0-9]\ [0-9][0-9]\ [0-9][0-9]\ \-\ [0-9]{1}",
]

key_extraction_section_regex = [
    "[0-9][0-9][0-9][0-9][0-9][0-9]",
    "[0-9][0-9] [0-9][0-9][0-9][0-9]",
    "[0-9][0-9] [0-9][0-9] [0-9][0-9]",
    "[0-9][0-9][0-9][0-9][0-9]"
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
    for pattern in section_extraction_regex:
        discovered_match = re.findall(pattern, text)

        if discovered_match:
            temp_list += discovered_match

    return temp_list


# extract the sections for the keys
def map_sections_to_page(sliced_text, page_num, section_candidates):
    for block in sliced_text:
        line = block[4]
        for pattern in key_extraction_section_regex:
            discovered_match = re.findall(pattern, line)
            if discovered_match:
                if hasattr(section_candidates, str(page_num)):
                    section_candidates[page_num] += discovered_match
                else:
                    section_candidates[page_num] = discovered_match
    return section_candidates


def normalize_section(section):
    return trim_section(section.replace(" ", ""))


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


def keyword_in_text(text, keywords):
    for keyword in keywords:
        pattern = r'\b' + re.escape(keyword.lower()) + r'\b'
        if re.search(pattern, text):
            return True
    return False


def read_and_filter_blocks(blocks, keywords):
    blocks_filtered = []
    for block in blocks:
        # print(block[1])
        if keyword_in_text(block[3].lower().translate(str.maketrans('', '', string.punctuation)), keywords[block[1]]):
            blocks_filtered.append(block)
    return blocks_filtered


def parse_data(test_df, train_data, model, sections):
    section_prediction_df = {}

    for index, section in enumerate(sections):
        train_data_temp = train_data[train_data.section == section].copy()
        train_data_sentences = train_data_temp["highlight"].tolist()
        train_model_embed = model.encode(train_data_sentences, convert_to_tensor=True)

        predict_content = test_df[test_df.section == section].copy()

        predict_sentence = predict_content["text"].tolist()

        if len(predict_sentence):
            predict_model_embed = model.encode(predict_sentence, convert_to_tensor=True)
            scores = util.cos_sim(predict_model_embed, train_model_embed)

            predict_content["score"] = np.array(scores.cpu()).max(axis=1)

            section_prediction_df[section] = predict_content

    return section_prediction_df


def parse_data_electrical_contractor(test_data, train_data, group_train_data, delete_train_data, model):
    predictions = []
    highlighted_train_data = train_data[train_data["Highlighted"] == 1]
    train_data_sentences = highlighted_train_data["Text"].tolist()
    train_model_embed = model.encode(train_data_sentences, convert_to_tensor=True)

    highlighted_train_data = train_data[train_data["Highlighted"] == 1]
    train_data_sentences = highlighted_train_data["Text"].tolist()
    train_model_embed = model.encode(train_data_sentences, convert_to_tensor=True)

    group_train_data = group_train_data[group_train_data["Highlighted"] == 1]
    group_train_data_sentences = group_train_data["Text"].tolist()
    group_train_model_embed = model.encode(group_train_data_sentences, convert_to_tensor=True)

    for text in test_data:
        if len(text[2]):
            predict_model_embed = model.encode(text[2], convert_to_tensor=True)
            scores = util.cos_sim(predict_model_embed, train_model_embed)
            scores_np = np.array(scores.cpu())
            max_score = scores_np.max()

            scores = util.cos_sim(predict_model_embed, group_train_model_embed)
            scores_np = np.array(scores.cpu())
            max_index = scores_np.argmax()

            closest_match_group = group_train_data.iloc[max_index]["Group"]

            predictions.append({
                'text': text[2],
                'quads': text[1],
                'section': '26051',
                'group': closest_match_group,
                'score': max_score
            })

    return predictions


def parse_data_general_contractor(test_data, train_data, delete_train_data, model):
    predictions = []
    highlighted_train_data = train_data[train_data["Highlighted"] == 1]
    train_data_sentences = highlighted_train_data["Text"].tolist()
    train_model_embed = model.encode(train_data_sentences, convert_to_tensor=True)

    delete_train_data_sentences = delete_train_data["text"].tolist()
    delete_train_model_embed = model.encode(delete_train_data_sentences, convert_to_tensor=True)

    for text in test_data:
        if len(text[3]):
            predict_model_embed = model.encode(text[3], convert_to_tensor=True)
            scores = util.cos_sim(predict_model_embed, train_model_embed)
            scores_np = np.array(scores.cpu())
            max_score = scores_np.max()

            delete_scores = util.cos_sim(predict_model_embed, delete_train_model_embed)
            delete_max_score = np.array(delete_scores.cpu()).max()

            predictions.append({
                'text': text[3],
                'quads': text[2],
                'section': text[0],
                'group': text[1],
                'score': max_score,
                'delete_score': delete_max_score
            })

    return predictions


def append_if_not_ending(main_str, suffix):
    if not main_str.endswith(suffix):
        main_str += suffix
    return main_str


def extract_division_26_blocks(spec):
    section_26_blocks = set()
    section_26_blocks_df = []
    doc = spec

    section_26_started = False
    current_sections = []
    current_sections = {'text': [], 'quad': None, 'section': '260501'}  # Initialize as a dictionary
    for page_num in range(doc.page_count):
        page = doc.load_page(page_num)
        blocks = page.get_text("blocks")

        for block in blocks:
            text = block[4].strip()
            current_sections['quad'] = block[:4]

            if text.startswith("PART") or text.startswith("DIVISION"):
                current_sections['text'] = [text]
            elif re.match(r"^\d+\.\d+", text):  # Matches sections like "2.1"
                current_sections['text'] = current_sections['text'][:1] + [text]
            elif re.match(r"^[A-Z]\.", text):  # Matches subsections like "A."
                current_sections['text'] = current_sections['text'][:2] + [text]
            elif re.match(r"^\d+\.", text):  # Matches numbered lists like "1."
                current_sections['text'] = current_sections['text'][:3] + [text]
            elif re.match(r"^[a-z]\.", text):  # Matches lowercase subsections like "a."
                current_sections['text'] = current_sections['text'][:3] + [text]

            if re.search(r"\bDIVISION\s*26\b", text, re.IGNORECASE) or re.match(r"^\s*26\b", text):
                section_26_started = True
            elif section_26_started and re.search(r"\bDIVISION\s*\d+\b", text, re.IGNORECASE):
                section_26_started = False

            if section_26_started and append_if_not_ending(" > ".join(current_sections), text):
                section_26_blocks.add(('26051', str(current_sections['quad']), " > ".join(current_sections['text'])))

    return list(section_26_blocks)


def extract_division_1_blocks(spec, section, group):
    section_01_blocks = set()
    doc = spec

    current_sections = {'text': [], 'quad': None, 'section': section}  # Initialize as a dictionary
    for page_num in range(doc.page_count):
        page = doc.load_page(page_num)

        # Extract the bottom portion of the page
        page_height = page.rect.height
        header_region = fitz.Rect(0, 0, page.rect.width, 100)  # Assuming header is in top 100 pixels
        header_text = page.get_text("text", clip=header_region).strip()
        footer_region = fitz.Rect(0, page_height - 100, page.rect.width, page_height)
        footer_text = page.get_text("text", clip=footer_region)
        if section in header_text.replace(' ', '') or section in footer_text.replace(' ', ''):
            blocks = page.get_text("blocks")
            for block in blocks:
                text = block[4].strip()
                current_sections['quad'] = block[:4]  # Update the 'quad' key for each block

                if text.startswith("PART") or text.startswith("DIVISION"):
                    current_sections['text'] = [text]
                elif re.match(r"^\d+\.\d+", text):  # Matches sections like "2.1"
                    current_sections['text'] = current_sections['text'][:1] + [text]
                elif re.match(r"^[A-Z]\.", text):  # Matches subsections like "A."
                    current_sections['text'] = current_sections['text'][:2] + [text]
                elif re.match(r"^\d+\.", text):  # Matches numbered lists like "1."
                    current_sections['text'] = current_sections['text'][:3] + [text]
                elif re.match(r"^[a-z]\.", text):  # Matches lowercase subsections like "a."
                    current_sections['text'] = current_sections['text'][:3] + [text]

                # Add the tuple of dictionary values to the set
                if append_if_not_ending(" > ".join(current_sections['text']), text):
                    section_01_blocks.add((current_sections['section'], group, str(current_sections['quad']), " > ".join(current_sections['text'])))

    return list(section_01_blocks)

