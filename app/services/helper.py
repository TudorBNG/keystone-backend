import re
import pandas as pd
from sentence_transformers import util
import numpy as np
import fitz
import string

section_regex = [
    "[0-9][0-9][0-9][0-9][0-9]"
    "[0-9][0-9][0-9][0-9][0-9][0-9]",
    "[0-9][0-9] [0-9][0-9][0-9][0-9]",
    "[0-9][0-9] [0-9][0-9] [0-9][0-9]",
]


def map_sections_to_page(text, page_num, section_candidates):
    sliced_text = text[-3:][::-1]
    for block in sliced_text:
        line = block[4]
        for pattern in section_regex:
            discovered_match = re.findall(pattern, line)
            if discovered_match:
                section_candidates[page_num] += discovered_match

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


def keyword_in_text(text, keywords):
    for keyword in keywords:
        pattern = r'\b' + re.escape(keyword.lower()) + r'\b'
        if re.search(pattern, text):
            return True
    return False


def read_and_filter_blocks(blocks):
    KEYWORDS = ['security']
    blocks_filtered = []
    for block in blocks:
        if keyword_in_text(block.lower().translate(str.maketrans('', '', string.punctuation)), KEYWORDS):
            blocks_filtered.append(block)


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
        if len(text):
            predict_model_embed = model.encode(text, convert_to_tensor=True)
            scores = util.cos_sim(predict_model_embed, train_model_embed)
            scores_np = np.array(scores.cpu())
            max_score = scores_np.max()

            scores = util.cos_sim(predict_model_embed, group_train_model_embed)
            scores_np = np.array(scores.cpu())
            max_index = scores_np.argmax()

            closest_match_group = group_train_data.iloc[max_index]["Group"]

            predictions.append({
                'text': text,
                'score': max_score,
                'group': closest_match_group
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
        if len(text):
            predict_model_embed = model.encode(text, convert_to_tensor=True)
            scores = util.cos_sim(predict_model_embed, train_model_embed)
            scores_np = np.array(scores.cpu())
            max_score = scores_np.max()

            delete_scores = util.cos_sim(predict_model_embed, delete_train_model_embed)
            delete_max_score = np.array(delete_scores.cpu()).max()

            predictions.append({
                'text': text,
                'score': max_score,
                'delete_score': delete_max_score
            })

    return predictions


def append_if_not_ending(main_str, suffix):
    if not main_str.endswith(suffix):
        main_str += suffix
    return main_str


def extract_division_26_blocks(pdf_path):
    section_26_blocks = set()
    section_26_blocks_df = []
    doc = fitz.open(pdf_path)

    section_26_started = False
    current_sections = []
    for page_num in range(doc.page_count):
        page = doc.load_page(page_num)
        blocks = page.get_text("blocks")

        for block in blocks:
            text = block[4].strip()
            if text.startswith("PART") or text.startswith("DIVISION"):
                current_sections = [text]
            elif re.match(r"^\d+\.\d+", text):  # Matches sections like "2.1"
                current_sections = current_sections[:1] + [text]
            elif re.match(r"^[A-Z]\.", text):  # Matches subsections like "A."
                current_sections = current_sections[:2] + [text]
            elif re.match(r"^\d+\.", text):  # Matches numbered lists like "1."
                current_sections = current_sections[:3] + [text]
            elif re.match(r"^[a-z]\.", text):  # Matches lowercase subsections like "a."
                current_sections = current_sections[:3] + [text]

            if re.search(r"\bDIVISION\s*26\b", text, re.IGNORECASE) or re.match(r"^\s*26\b", text):
                section_26_started = True
            elif section_26_started and re.search(r"\bDIVISION\s*\d+\b", text, re.IGNORECASE):
                section_26_started = False

            if section_26_started and append_if_not_ending(" > ".join(current_sections), text):
                section_26_blocks.add(append_if_not_ending(" > ".join(current_sections), text))

    return list(section_26_blocks)


def extract_division_1_blocks(pdf_path, section):
    section_01_blocks = set()
    section_01_blocks_df = []
    doc = fitz.open(pdf_path)

    current_sections = []
    for page_num in range(doc.page_count):
        page = doc.load_page(page_num)

        # Extract the bottom portion of the page
        page_height = page.rect.height
        header_region = fitz.Rect(0, 0, page.rect.width, 100)  # Assuming header is in top 100 pixels
        header_text = page.get_text("text", clip=header_region).strip()
        footer_region = fitz.Rect(0, page_height - 100, page.rect.width, page_height)
        footer_text = page.get_text("text", clip=footer_region)

        if section in header_text.replace(' ', '') or section in footer_text.replace(' ', ''):
            # print('reached here')
            blocks = page.get_text("blocks")
            for block in blocks:
                text = block[4].strip()
                if text.startswith("PART") or text.startswith("DIVISION"):
                    current_sections = [text]
                elif re.match(r"^\d+\.\d+", text):  # Matches sections like "2.1"
                    current_sections = current_sections[:1] + [text]
                elif re.match(r"^[A-Z]\.", text):  # Matches subsections like "A."
                    current_sections = current_sections[:2] + [text]
                elif re.match(r"^\d+\.", text):  # Matches numbered lists like "1."
                    current_sections = current_sections[:3] + [text]
                elif re.match(r"^[a-z]\.", text):  # Matches lowercase subsections like "a."
                    current_sections = current_sections[:3] + [text]

                if append_if_not_ending(" > ".join(current_sections), text):
                    section_01_blocks.add(append_if_not_ending(" > ".join(current_sections), text))

    return list(section_01_blocks)
