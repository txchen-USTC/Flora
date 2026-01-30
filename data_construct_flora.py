import json
import random
import argparse
from tqdm import tqdm

import numpy as np
from transformers import AutoTokenizer
import re

from flora_formats_rules import *

def get_random_indicators(args, forced_overall_indicator=None):
    if forced_overall_indicator is not None:
        overall_indicator = forced_overall_indicator
    elif args.version == 'format':
        overall_indicator = 0
    elif args.version == 'order':
        overall_indicator = random.randint(0, 1)
    elif args.version == 'mask':
        overall_indicator = random.randint(0, 1)
        overall_indicator = 2 if overall_indicator == 1 else overall_indicator
    elif args.version == 'both':
        overall_indicator = random.randint(0, 2)
    elif args.version == 'all':
        overall_indicator = random.randint(0, 6)
    else:
        raise ValueError("Invalid version argument")

    # Rest of the function remains unchanged...

    # 0 for format only, 1 for format + order, 2 for format + mask, 3 for format + ID ANSWER, 4 for format + ANSWER_NO_ANSWER, 
    # 5 for format + QA_FEWSHOT_ANSWER, 6 for format + ANSWER_BEFORE_AFTER

    ################################################
    # FORMAT VERSION
    ################################################

    digit_tag_idx = random.randint(0, len(DIGITS_TAG_LIST)-1)
    digit_punc_idx = random.randint(0, len(DIGITS_PUNC_LIST)-1)
    begin_end_tag_punc_idx = random.randint(0, len(BEGIN_END_TAG_PUNC_LIST)-1)
    begin_end_text_idx = random.randint(0, len(BEGIN_END_TEXT_LIST)-1)
    format_instruction_interval_tag_idx = random.randint(0, len(FORMAT_INSTRUCTION_INTERVAL_TAG_LIST)-1)
    fomat_meta_instruction_idx = random.randint(0, len(FORMAT_META_INSTRUCTION_LIST)-1)
    fomat_meta_instruction_no_idx = random.randint(0, len(FORMAT_META_INSTRUCTION_NO_LIST)-1)
    format_overall_instruction_idx = random.randint(0, len(FORMAT_OVERALL_INSTRUCTION_LIST)-1)

    newline_for_digit = random.choice([True, False])
    lowercase_for_begin_end_text = random.choice([True, False])
    do_begin_end_format = random.choice([True, False])
    do_no_begin_end_format_no_ins = random.choice([True, False])

    ################################################
    # ORDER VERSION
    ################################################

    order_method_idx = random.randint(0, len(ORDER_METHOD_INSTRUCTION_PAIR_DICT)-1)
    order_method = list(ORDER_METHOD_INSTRUCTION_PAIR_DICT.keys())[order_method_idx]
    order_meta_instruction_idx = random.randint(0, len(ORDER_METHOD_INSTRUCTION_PAIR_DICT[order_method])-1)
    order_overall_instruction_idx = random.randint(0, len(ORDER_OVERALL_INSTRUCTION_LIST)-1)
    order_seq_tag_idx = random.randint(0, len(ORDER_SEQ_TAG_LIST)-1)
    order_seq_interval_idx = random.randint(0, len(ORDER_SEQ_INTERVAL_LIST)-1)

    order_fixed_list = [i for i in range(MAX_INSTRUCTION_NUMBER)]
    random.shuffle(order_fixed_list)

    ################################################
    # MASk VERSION
    ################################################

    mask_method_idx = random.randint(0, len(MASK_METHOD_INSTRUCTION_PAIR_DICT)-1)
    mask_method = list(MASK_METHOD_INSTRUCTION_PAIR_DICT.keys())[mask_method_idx]
    mask_meta_instruction_idx = random.randint(0, len(MASK_METHOD_INSTRUCTION_PAIR_DICT[mask_method])-1)
    mask_overall_instruction_idx = random.randint(0, len(MASK_OVERALL_INSTRUCTION_LIST)-1)
    mask_seq_tag_idx = random.randint(0, len(MASK_SEQ_TAG_LIST)-1)
    mask_seq_interval_idx = random.randint(0, len(MASK_SEQ_INTERVAL_LIST)-1)

    mask_ratio = random.uniform(MASK_RATIO_RANGE[0], MASK_RATIO_RANGE[1])
    mask_fixed_list = [i for i in range(MAX_INSTRUCTION_NUMBER)]
    random.shuffle(mask_fixed_list)

    ################################################
    # ID ANSWER VERSION
    ################################################

    ID_ANSWER_method_idx = random.randint(0, len(ID_ANSWER_METHOD_INSTRUCTION_PAIR_DICT)-1)
    ID_ANSWER_method = list(ID_ANSWER_METHOD_INSTRUCTION_PAIR_DICT.keys())[ID_ANSWER_method_idx]
    ID_ANSWER_meta_instruction_idx = random.randint(0, len(ID_ANSWER_METHOD_INSTRUCTION_PAIR_DICT[ID_ANSWER_method])-1)
    ID_ANSWER_overall_instruction_idx = random.randint(0, len(ID_ANSWER_OVERALL_INSTRUCTION_LIST)-1)
    ID_ANSWER_seq_tag_idx = random.randint(0, len(ID_ANSWER_SEQ_TAG_LIST)-1)
    ID_ANSWER_seq_interval_idx = random.randint(0, len(ID_ANSWER_SEQ_INTERVAL_LIST)-1)

    ID_ANSWER_fixed_list = [i for i in range(MAX_INSTRUCTION_NUMBER)]
    random.shuffle(ID_ANSWER_fixed_list)

    ################################################
    # ANSWER_NO_ANSWER VERSION
    ################################################

    ANSWER_NO_ANSWER_method_idx = random.randint(0, len(ANSWER_NO_ANSWER_METHOD_INSTRUCTION_PAIR_DICT)-1)
    ANSWER_NO_ANSWER_method = list(ANSWER_NO_ANSWER_METHOD_INSTRUCTION_PAIR_DICT.keys())[ANSWER_NO_ANSWER_method_idx]
    ANSWER_NO_ANSWER_meta_instruction_idx = random.randint(0, len(ANSWER_NO_ANSWER_METHOD_INSTRUCTION_PAIR_DICT[ANSWER_NO_ANSWER_method])-1)
    ANSWER_NO_ANSWER_overall_instruction_idx = random.randint(0, len(ANSWER_NO_ANSWER_OVERALL_INSTRUCTION_LIST)-1)
    ANSWER_NO_ANSWER_seq_tag_idx = random.randint(0, len(ANSWER_NO_ANSWER_SEQ_TAG_LIST)-1)
    ANSWER_NO_ANSWER_seq_interval_idx = random.randint(0, len(ANSWER_NO_ANSWER_SEQ_INTERVAL_LIST)-1)

    ANSWER_NO_ANSWER_ratio = random.uniform(MASK_RATIO_RANGE[0], MASK_RATIO_RANGE[1])
    ANSWER_NO_ANSWER_fixed_list = [i for i in range(MAX_INSTRUCTION_NUMBER)]
    random.shuffle(ANSWER_NO_ANSWER_fixed_list)

    ################################################
    # QA_FEWSHOT_ANSWER VERSION
    ################################################

    QA_FEWSHOT_ANSWER_method_idx = random.randint(0, len(QA_FEWSHOT_ANSWER_METHOD_INSTRUCTION_PAIR_DICT)-1)
    QA_FEWSHOT_ANSWER_method = list(QA_FEWSHOT_ANSWER_METHOD_INSTRUCTION_PAIR_DICT.keys())[QA_FEWSHOT_ANSWER_method_idx]
    QA_FEWSHOT_ANSWER_meta_instruction_idx = random.randint(0, len(QA_FEWSHOT_ANSWER_METHOD_INSTRUCTION_PAIR_DICT[QA_FEWSHOT_ANSWER_method])-1)
    QA_FEWSHOT_ANSWER_overall_instruction_idx = random.randint(0, len(QA_FEWSHOT_ANSWER_OVERALL_INSTRUCTION_LIST)-1)
    QA_FEWSHOT_ANSWER_seq_tag_idx = random.randint(0, len(QA_FEWSHOT_ANSWER_SEQ_TAG_LIST)-1)
    QA_FEWSHOT_ANSWER_seq_interval_idx = random.randint(0, len(QA_FEWSHOT_ANSWER_SEQ_INTERVAL_LIST)-1)

    QA_FEWSHOT_ANSWER_ratio = random.uniform(MASK_RATIO_RANGE[0], MASK_RATIO_RANGE[1])
    QA_FEWSHOT_ANSWER_fixed_list = [i for i in range(MAX_INSTRUCTION_NUMBER)]
    random.shuffle(QA_FEWSHOT_ANSWER_fixed_list)

    ################################################
    # NEXT_LINE_COMPLETE VERSION
    ################################################

    NEXT_LINE_COMPLETE_method_idx = random.randint(0, len(NEXT_LINE_COMPLETE_INSTRUCTION_PAIR_DICT)-1)
    NEXT_LINE_COMPLETE_method = list(NEXT_LINE_COMPLETE_INSTRUCTION_PAIR_DICT.keys())[NEXT_LINE_COMPLETE_method_idx]
    NEXT_LINE_COMPLETE_meta_instruction_idx = random.randint(0, len(NEXT_LINE_COMPLETE_INSTRUCTION_PAIR_DICT[NEXT_LINE_COMPLETE_method])-1)
    NEXT_LINE_COMPLETE_overall_instruction_idx = random.randint(0, len(NEXT_LINE_COMPLETE_OVERALL_INSTRUCTION_LIST)-1)
    NEXT_LINE_COMPLETE_seq_tag_idx = random.randint(0, len(NEXT_LINE_COMPLETE_SEQ_TAG_LIST)-1)
    NEXT_LINE_COMPLETE_seq_interval_idx = random.randint(0, len(NEXT_LINE_COMPLETE_SEQ_INTERVAL_LIST)-1)

    NEXT_LINE_COMPLETE_ratio = random.uniform(MASK_RATIO_RANGE[0], MASK_RATIO_RANGE[1])
    NEXT_LINE_COMPLETE_fixed_list = [i for i in range(MAX_INSTRUCTION_NUMBER)]
    random.shuffle(NEXT_LINE_COMPLETE_fixed_list)


    ################################################
    # ANSWER_BEFORE_AFTER VERSION
    ################################################

    ANSWER_BEFORE_AFTER_method_idx = random.randint(0, len(ANSWER_BEFORE_AFTER_INSTRUCTION_PAIR_DICT)-1)
    ANSWER_BEFORE_AFTER_method = list(ANSWER_BEFORE_AFTER_INSTRUCTION_PAIR_DICT.keys())[ANSWER_BEFORE_AFTER_method_idx]
    ANSWER_BEFORE_AFTER_meta_instruction_idx = random.randint(0, len(ANSWER_BEFORE_AFTER_INSTRUCTION_PAIR_DICT[ANSWER_BEFORE_AFTER_method])-1)
    ANSWER_BEFORE_AFTER_overall_instruction_idx = random.randint(0, len(ANSWER_BEFORE_AFTER_OVERALL_INSTRUCTION_LIST)-1)
    ANSWER_BEFORE_AFTER_seq_tag_idx = random.randint(0, len(ANSWER_BEFORE_AFTER_SEQ_TAG_LIST)-1)
    ANSWER_BEFORE_AFTER_seq_interval_idx = random.randint(0, len(ANSWER_BEFORE_AFTER_SEQ_INTERVAL_LIST)-1)

    ANSWER_BEFORE_AFTER_ratio = random.uniform(MASK_RATIO_RANGE[0], MASK_RATIO_RANGE[1])
    ANSWER_BEFORE_AFTER_fixed_list = [i for i in range(MAX_INSTRUCTION_NUMBER)]
    random.shuffle(ANSWER_BEFORE_AFTER_fixed_list)

    # Wrap all indicators
    random_indicators = {
        'overall_indicator':overall_indicator,
        # Format version
        'digit_tag_idx':digit_tag_idx,
        'digit_punc_idx':digit_punc_idx,
        'begin_end_tag_punc_idx':begin_end_tag_punc_idx,
        'begin_end_text_idx':begin_end_text_idx,
        'format_instruction_interval_tag_idx':format_instruction_interval_tag_idx,
        'fomat_meta_instruction_idx':fomat_meta_instruction_idx,
        'fomat_meta_instruction_no_idx':fomat_meta_instruction_no_idx,
        'format_overall_instruction_idx':format_overall_instruction_idx,
        'newline_for_digit':newline_for_digit,
        'lowercase_for_begin_end_text':lowercase_for_begin_end_text,
        'do_begin_end_format':do_begin_end_format,
        'do_no_begin_end_format_no_ins':do_no_begin_end_format_no_ins,

        # Order version
        'order_method_idx':order_method_idx,
        'order_meta_instruction_idx':order_meta_instruction_idx,
        'order_overall_instruction_idx':order_overall_instruction_idx, 
        'order_fixed_list':order_fixed_list,
        'order_seq_tag_idx':order_seq_tag_idx,
        'order_seq_interval_idx':order_seq_interval_idx,

        # Mask version
        'mask_method_idx':mask_method_idx,
        'mask_meta_instruction_idx':mask_meta_instruction_idx,
        'mask_overall_instruction_idx':mask_overall_instruction_idx,
        'mask_ratio':mask_ratio,
        'mask_fixed_list':mask_fixed_list,
        'mask_seq_tag_idx':mask_seq_tag_idx,
        'mask_seq_interval_idx':mask_seq_interval_idx,

        # ID_ANSWER version
        'ID_ANSWER_method_idx':ID_ANSWER_method_idx,
        'ID_ANSWER_meta_instruction_idx':ID_ANSWER_meta_instruction_idx,
        'ID_ANSWER_overall_instruction_idx':ID_ANSWER_overall_instruction_idx, 
        'ID_ANSWER_fixed_list':ID_ANSWER_fixed_list,
        'ID_ANSWER_seq_tag_idx':ID_ANSWER_seq_tag_idx,
        'ID_ANSWER_seq_interval_idx':ID_ANSWER_seq_interval_idx,

        # ANSWER_NO_ANSWER version
        'ANSWER_NO_ANSWER_method_idx':ANSWER_NO_ANSWER_method_idx,
        'ANSWER_NO_ANSWER_meta_instruction_idx':ANSWER_NO_ANSWER_meta_instruction_idx,
        'ANSWER_NO_ANSWER_overall_instruction_idx':ANSWER_NO_ANSWER_overall_instruction_idx,
        'ANSWER_NO_ANSWER_ratio':ANSWER_NO_ANSWER_ratio,
        'ANSWER_NO_ANSWER_fixed_list':ANSWER_NO_ANSWER_fixed_list,
        'ANSWER_NO_ANSWER_seq_tag_idx':ANSWER_NO_ANSWER_seq_tag_idx,
        'ANSWER_NO_ANSWER_seq_interval_idx':ANSWER_NO_ANSWER_seq_interval_idx,

        # QA_FEWSHOT_ANSWER version
        'QA_FEWSHOT_ANSWER_method_idx':QA_FEWSHOT_ANSWER_method_idx,
        'QA_FEWSHOT_ANSWER_meta_instruction_idx':QA_FEWSHOT_ANSWER_meta_instruction_idx,
        'QA_FEWSHOT_ANSWER_overall_instruction_idx':QA_FEWSHOT_ANSWER_overall_instruction_idx,
        'QA_FEWSHOT_ANSWER_ratio':QA_FEWSHOT_ANSWER_ratio,
        'QA_FEWSHOT_ANSWER_fixed_list':QA_FEWSHOT_ANSWER_fixed_list,
        'QA_FEWSHOT_ANSWER_seq_tag_idx':QA_FEWSHOT_ANSWER_seq_tag_idx,
        'QA_FEWSHOT_ANSWER_seq_interval_idx':QA_FEWSHOT_ANSWER_seq_interval_idx,

        # NEXT_LINE_COMPLETE version
        'NEXT_LINE_COMPLETE_method_idx':NEXT_LINE_COMPLETE_method_idx,
        'NEXT_LINE_COMPLETE_meta_instruction_idx':NEXT_LINE_COMPLETE_meta_instruction_idx,
        'NEXT_LINE_COMPLETE_overall_instruction_idx':NEXT_LINE_COMPLETE_overall_instruction_idx,
        'NEXT_LINE_COMPLETE_ratio':NEXT_LINE_COMPLETE_ratio,
        'NEXT_LINE_COMPLETE_fixed_list':NEXT_LINE_COMPLETE_fixed_list,
        'NEXT_LINE_COMPLETE_seq_tag_idx':NEXT_LINE_COMPLETE_seq_tag_idx,
        'NEXT_LINE_COMPLETE_seq_interval_idx':NEXT_LINE_COMPLETE_seq_interval_idx,

        # ANSWER_BEFORE_AFTER version
        'ANSWER_BEFORE_AFTER_method_idx':ANSWER_BEFORE_AFTER_method_idx,
        'ANSWER_BEFORE_AFTER_meta_instruction_idx':ANSWER_BEFORE_AFTER_meta_instruction_idx,
        'ANSWER_BEFORE_AFTER_overall_instruction_idx':ANSWER_BEFORE_AFTER_overall_instruction_idx,
        'ANSWER_BEFORE_AFTER_ratio':ANSWER_BEFORE_AFTER_ratio,   
        'ANSWER_BEFORE_AFTER_fixed_list':ANSWER_BEFORE_AFTER_fixed_list,
        'ANSWER_BEFORE_AFTER_seq_tag_idx':ANSWER_BEFORE_AFTER_seq_tag_idx,
        'ANSWER_BEFORE_AFTER_seq_interval_idx':ANSWER_BEFORE_AFTER_seq_interval_idx,

    }

    return random_indicators

# Format only
def get_formated_data(random_indicators, used_items_temp):

    # Build instruction format
    if random_indicators['digit_tag_idx'] == 0:
        instruction_digit = ''
    else:
        instruction_digit = DIGITS_TAG_LIST[random_indicators['digit_tag_idx']] + DIGITS_PUNC_LIST[random_indicators['digit_punc_idx']]
        instruction_digit = instruction_digit + '\n' if random_indicators['newline_for_digit'] else instruction_digit
    
    # Build response format
    if random_indicators['do_begin_end_format']:

        # Get begin and end tag for instructions and responses
        begin_text, end_text = BEGIN_END_TEXT_LIST[random_indicators['begin_end_text_idx']][0], BEGIN_END_TEXT_LIST[random_indicators['begin_end_text_idx']][1]
        begin_text = begin_text.lower() if random_indicators['lowercase_for_begin_end_text'] else begin_text
        end_text = end_text.lower() if random_indicators['lowercase_for_begin_end_text'] else end_text

        begin_tag = BEGIN_END_TAG_PUNC_LIST[random_indicators['begin_end_tag_punc_idx']].format(text=begin_text)
        end_tag = BEGIN_END_TAG_PUNC_LIST[random_indicators['begin_end_tag_punc_idx']].format(text=end_text)

        # Get meta instruction
        meta_instruction = FORMAT_META_INSTRUCTION_LIST[random_indicators['fomat_meta_instruction_idx']]
        meta_instruction = meta_instruction.format(begin_tag=begin_tag, end_tag=end_tag)

        pass
    else:
        begin_tag, end_tag = '', ''

        # Get meta instruction
        if random_indicators['do_no_begin_end_format_no_ins']:
            meta_instruction = ''
        else:
            meta_instruction = FORMAT_META_INSTRUCTION_NO_LIST[random_indicators['fomat_meta_instruction_no_idx']]

    # Get overall instruction
    overall_instruction = FORMAT_OVERALL_INSTRUCTION_LIST[random_indicators['format_overall_instruction_idx']]

    instruction_all = ''
    response_all = ''
    # Process real data
    for i, data_i in enumerate(used_items_temp):
        instruction_i = data_i['instruction'] + '\n' + data_i['input'] if data_i['input'] != '' else data_i['instruction']
        response_i = data_i['output']

        # Get formated instruction_i
        instruction_digit_i = instruction_digit.format(i=i+1)
        instruction_i = instruction_digit_i + instruction_i

        # Get formated response_i
        response_i = instruction_digit_i + begin_tag + response_i + end_tag

        instruction_all += instruction_i
        response_all += response_i

        if i != len(used_items_temp) - 1:
            instruction_all += FORMAT_INSTRUCTION_INTERVAL_TAG_LIST[random_indicators['format_instruction_interval_tag_idx']]
            response_all += '\n\n\n'

        pass

    # Get real overall instruction
    overall_instruction = overall_instruction.format(meta_instruction=meta_instruction,instruction_all=instruction_all)

    return {'instruction':overall_instruction, 'output':response_all}

def get_ordered_response_list(random_indicators, instruction_list_ori, response_list_formated):

    order_method = list(ORDER_METHOD_INSTRUCTION_PAIR_DICT.keys())[random_indicators['order_method_idx']]

    if order_method == 'FIX':
        # Use the fix order
        order_fixed_list = random_indicators['order_fixed_list']
        # Remove the numbers greater than the length of the instruction list
        order_fixed_list = [i for i in order_fixed_list if i < len(instruction_list_ori)]
        ordered_response_list = [response_list_formated[i] for i in order_fixed_list]
    elif order_method == 'REVERSE':
        # Reverse the order
        ordered_response_list = response_list_formated[::-1]
    elif order_method == 'ALPHA':
        # Sort according to the first letter of each instruction
        ordered_response_list = [response for _, response in sorted(zip(instruction_list_ori, response_list_formated), key=lambda pair: pair[0])]
    elif order_method == 'REVERSE_ALPHA':
        # Sort according to the first letter of each instruction, reverse
        ordered_response_list = [response for _, response in sorted(zip(instruction_list_ori, response_list_formated), key=lambda pair: pair[0], reverse=True)]
    elif order_method == 'LENGTH_WORD':
        # Sort according to the word length of each instruction
        ordered_response_list = [response for _, response in sorted(zip(instruction_list_ori, response_list_formated), key=lambda pair: len(pair[0].split()))]
    elif order_method == 'REVERSE_LENGTH_WORD':
        # Sort according to the word length of each instruction, reverse
        ordered_response_list = [response for _, response in sorted(zip(instruction_list_ori, response_list_formated), key=lambda pair: len(pair[0].split()), reverse=True)]
    elif order_method == 'LENGTH_CHAR':
        # Sort according to the character length of each instruction
        ordered_response_list = [response for _, response in sorted(zip(instruction_list_ori, response_list_formated), key=lambda pair: len(pair[0]))]
    elif order_method == 'REVERSE_LENGTH_CHAR':
        # Sort according to the character length of each instruction, reverse
        ordered_response_list = [response for _, response in sorted(zip(instruction_list_ori, response_list_formated), key=lambda pair: len(pair[0]), reverse=True)]
    elif order_method == 'ODD_EVEN':
        # First respond to the odd-numbered instructions, then the even-numbered ones, Note: i startes from 0
        ordered_response_list = []
        for i, response in enumerate(response_list_formated):
            if i % 2 == 0:
                ordered_response_list.append(response)
        for i, response in enumerate(response_list_formated):
            if i % 2 != 0:
                ordered_response_list.append(response)
    elif order_method == 'EVEN_ODD':
        # First respond to the even-numbered instructions, then the odd-numbered ones, Note: i startes from 0
        ordered_response_list = []
        for i, response in enumerate(response_list_formated):
            if i % 2 != 0:
                ordered_response_list.append(response)
        for i, response in enumerate(response_list_formated):
            if i % 2 == 0:
                ordered_response_list.append(response)
    '''elif order_method == 'ANSWER_TO_INSTRUCTION_ID':
        
        ordered_response_list = []'''
    
    return ordered_response_list

def extract_last_line_of_code(output):
    # Match the code block enclosed in triple backticks
    matches = re.findall(r'```(.*?)```', output, re.DOTALL)
    if not matches:
        # Try matching single backticks if no triple backticks found
        matches = re.findall(r'`(.*?)`', output, re.DOTALL)

    if matches:
        # Take the first match (assuming there's only one code block per output)
        code_block = matches[0]
        # Split the code block by lines and take the last line
        last_line = code_block.split('\n')[-1].strip()
        return last_line

    return None

def get_masked_response_list(random_indicators, instruction_list_ori, response_list_formated):

    mask_method = list(MASK_METHOD_INSTRUCTION_PAIR_DICT.keys())[random_indicators['mask_method_idx']]

    if mask_method == 'NEXT_LINE':
        masked_response_list = []
        removed_last_sentences = []

        for response_i in response_list_formated:
            # Extract the last line of code if present
            last_line_of_code = extract_last_line_of_code(response_i)

            # Find the second last sentence boundary
            matches = list(re.finditer(r'\. ', response_i))
            if len(matches) > 1:
                second_last_sentence_boundary = matches[-2]
                # Split the response into two parts: before and after the second last sentence boundary
                masked_response_i = response_i[:second_last_sentence_boundary.end()].strip()+' {}'
                last_sentence = response_i[second_last_sentence_boundary.end():].strip()
            else:
                # If there is only one sentence boundary or none, treat the entire response as the last sentence
                masked_response_i = ''
                last_sentence = response_i.strip()

            # If there was a code block, use its last line instead of the last sentence
            if last_line_of_code:
                last_sentence = last_line_of_code
                #masked_response_i = masked_response_i

            masked_response_list.append(masked_response_i)
            removed_last_sentences.append(last_sentence)

        return masked_response_list, removed_last_sentences
    
        '''for response_i in response_list_formated:
            # Find the second last sentence boundary
            matches = list(re.finditer(r'\. ', response_i))
            if len(matches) > 1:
                second_last_sentence_boundary = matches[-2]
                # Split the response into two parts: before and after the second last sentence boundary
                masked_response_i = response_i[:second_last_sentence_boundary.end()].strip()  # Include the period and space
                last_sentence = response_i[second_last_sentence_boundary.end():].strip()
            else:
                # If there is only one sentence boundary or none, treat the entire response as the last sentence
                masked_response_i = ''
                last_sentence = response_i.strip()

        
            masked_response_list.append(masked_response_i)
            removed_last_sentences.append(last_sentence)
    
        return masked_response_list, removed_last_sentences'''

    if mask_method == 'FIX':
        # Ignore the instructions in the fixed list
        mask_threshold = max(int(len(instruction_list_ori) * random_indicators['mask_ratio']),1)
        mask_fixed_list = random_indicators['mask_fixed_list']
        mask_fixed_list = [i for i in mask_fixed_list if i < len(instruction_list_ori)]
        masked_response_list = [response for i, response in enumerate(response_list_formated) if i not in mask_fixed_list[:mask_threshold]]
    
    elif mask_method == 'WORD_LONG':
        # Ignore the longest n instructions
        instruction_length_list = [len(instruction.split()) for instruction in instruction_list_ori]
        sorted_pairs = sorted(enumerate(instruction_length_list), key=lambda x: x[1], reverse=True)
        sorted_indices = [index for index, value in sorted_pairs]
        mask_threshold = max(int(len(instruction_list_ori) * random_indicators['mask_ratio']),1)
        mask_fixed_list = sorted_indices[:mask_threshold]
        masked_response_list = [response for i, response in enumerate(response_list_formated) if i not in mask_fixed_list[:mask_threshold]]
    elif mask_method == 'WORD_SHORT':
        # Ignore the shortest n instructions
        instruction_length_list = [len(instruction.split()) for instruction in instruction_list_ori]
        sorted_pairs = sorted(enumerate(instruction_length_list), key=lambda x: x[1], reverse=False)
        sorted_indices = [index for index, value in sorted_pairs]
        mask_threshold = max(int(len(instruction_list_ori) * random_indicators['mask_ratio']),1)
        mask_fixed_list = sorted_indices[:mask_threshold]
        masked_response_list = [response for i, response in enumerate(response_list_formated) if i not in mask_fixed_list[:mask_threshold]]
    elif mask_method == 'ODD':
        # Ignore the odd-numbered instructions, keep the even, Note: i startes from 0
        masked_response_list = [response for i, response in enumerate(response_list_formated) if i % 2 != 0]
    elif mask_method == 'EVEN':
        # Ignore the odd-numbered instructions, keep the odd, Note: i startes from 0
        masked_response_list = [response for i, response in enumerate(response_list_formated) if i % 2 == 0]
    '''elif mask_method == 'QA_FEWSHOT_ANSWER':
        masked_response_list = []
    elif mask_method == 'QA_mask':
        masked_response_list = []
        for i, response_i in enumerate(response_list_formated):
            if response_i!= '':  # only add responses that are not empty
                masked_response_list.append(response_i)
            else:
                masked_response_list.append('')'''

    return masked_response_list
    

def wrap_instruction_response(random_indicators, instruction_list_formated, response_list_formated):
    instruction_all = ''
    response_all = ''
    for i, instruction_i in enumerate(instruction_list_formated):
        instruction_all += instruction_i
        if i != len(instruction_list_formated) - 1:
            instruction_all += FORMAT_INSTRUCTION_INTERVAL_TAG_LIST[random_indicators['format_instruction_interval_tag_idx']]
    
    for i, response_i in enumerate(response_list_formated):
        response_all += response_i
        if i != len(response_list_formated) - 1:
            response_all += '\n\n'

    return instruction_all, response_all

# Format + order + mask
def get_formated_data_pro(random_indicators, used_items_temp):

    # Build instruction format
    if random_indicators['digit_tag_idx'] == 0:
        instruction_digit = ''
    else:
        instruction_digit = DIGITS_TAG_LIST[random_indicators['digit_tag_idx']] + DIGITS_PUNC_LIST[random_indicators['digit_punc_idx']]
        instruction_digit = instruction_digit + '\n' if random_indicators['newline_for_digit'] else instruction_digit
    
    # Build response format
    if random_indicators['do_begin_end_format']:

        # Get begin and end tag for instructions and responses
        begin_text, end_text = BEGIN_END_TEXT_LIST[random_indicators['begin_end_text_idx']][0], BEGIN_END_TEXT_LIST[random_indicators['begin_end_text_idx']][1]
        begin_text = begin_text.lower() if random_indicators['lowercase_for_begin_end_text'] else begin_text
        end_text = end_text.lower() if random_indicators['lowercase_for_begin_end_text'] else end_text

        begin_tag = BEGIN_END_TAG_PUNC_LIST[random_indicators['begin_end_tag_punc_idx']].format(text=begin_text)
        end_tag = BEGIN_END_TAG_PUNC_LIST[random_indicators['begin_end_tag_punc_idx']].format(text=end_text)

        # Get meta instruction
        meta_instruction = FORMAT_META_INSTRUCTION_LIST[random_indicators['fomat_meta_instruction_idx']]
        meta_instruction = meta_instruction.format(begin_tag=begin_tag, end_tag=end_tag)

        pass
    else:
        begin_tag, end_tag = '', ''

        # Get meta instruction
        if random_indicators['do_no_begin_end_format_no_ins']:
            meta_instruction = ''
        else:
            meta_instruction = FORMAT_META_INSTRUCTION_NO_LIST[random_indicators['fomat_meta_instruction_no_idx']]

    # Get overall instruction
    format_overall_instruction_format = FORMAT_OVERALL_INSTRUCTION_LIST[random_indicators['format_overall_instruction_idx']]

    instruction_list_ori = []
    instruction_list_formated = []
    response_list_formated = []
    

    for i, data_i in enumerate(used_items_temp):
        instruction_i = data_i['instruction'] + '\n' + data_i['input'] if data_i['input'] != '' else data_i['instruction']
        response_i = data_i['output']

        # Get formated instruction_i
        instruction_digit_i = instruction_digit.format(i=i+1)
        instruction_i_formated = instruction_digit_i + instruction_i

        # Get formated response_i
        response_i_formated = instruction_digit_i + begin_tag + response_i + end_tag

        instruction_list_ori.append(instruction_i)
        instruction_list_formated.append(instruction_i_formated)
        response_list_formated.append(response_i_formated)

    if random_indicators['overall_indicator'] == 0 or len(instruction_list_ori) == 1:
        # Format only
        # If there is only one instruction, we do not need to do the order or mask
        instruction_all, response_all = wrap_instruction_response(random_indicators, instruction_list_formated, response_list_formated)
        overall_instruction = format_overall_instruction_format.format(meta_instruction=meta_instruction,instruction_all=instruction_all)
    
    elif random_indicators['overall_indicator'] == 1:
        # Format + order
        order_method = list(ORDER_METHOD_INSTRUCTION_PAIR_DICT.keys())[random_indicators['order_method_idx']]
        ordered_response_list = get_ordered_response_list(random_indicators, instruction_list_ori, response_list_formated)
        #assert len(ordered_response_list) == len(response_list_formated)
        instruction_all, response_all = wrap_instruction_response(random_indicators, instruction_list_formated, ordered_response_list)

        order_meta_instruction = ORDER_METHOD_INSTRUCTION_PAIR_DICT[order_method][random_indicators['order_meta_instruction_idx']]
        order_overall_instruction_format = ORDER_OVERALL_INSTRUCTION_LIST[random_indicators['order_overall_instruction_idx']]

        if order_method == 'FIX':
            order_fixed_list = random_indicators['order_fixed_list']
            order_fixed_list = [i for i in order_fixed_list if i < len(instruction_list_ori)]
            seq = ''
            for idx_list, idx_real in enumerate(order_fixed_list):
                if idx_list != len(order_fixed_list) - 1:
                    seq = seq + str(idx_real+1) + ORDER_SEQ_INTERVAL_LIST[random_indicators['order_seq_interval_idx']]
                else:
                    seq = seq + str(idx_real+1)
            seq = ORDER_SEQ_TAG_LIST[random_indicators['order_seq_tag_idx']].format(i=seq)
            order_meta_instruction = order_meta_instruction.format(seq)
            pass

        instruction_all = format_overall_instruction_format.format(meta_instruction=meta_instruction,instruction_all=instruction_all)
        overall_instruction = order_overall_instruction_format.format(meta_instruction=order_meta_instruction,instruction_all=instruction_all)

    elif random_indicators['overall_indicator'] == 2:
        # Format + mask
        mask_method = list(MASK_METHOD_INSTRUCTION_PAIR_DICT.keys())[random_indicators['mask_method_idx']]

        
        
        masked_response_list = get_masked_response_list(random_indicators, instruction_list_ori, response_list_formated)
        instruction_all, response_all = wrap_instruction_response(random_indicators, instruction_list_formated, masked_response_list)

        mask_meta_instruction = MASK_METHOD_INSTRUCTION_PAIR_DICT[mask_method][random_indicators['mask_meta_instruction_idx']]
        mask_overall_instruction_format = MASK_OVERALL_INSTRUCTION_LIST[random_indicators['mask_overall_instruction_idx']]
        
    
        if mask_method == 'FIX':
        #if mask_method in ['FIX', 'FIX_REVERSE']:
            mask_threshold = max(int(len(instruction_list_ori) * random_indicators['mask_ratio']),1)
            mask_fixed_list = random_indicators['mask_fixed_list']
            mask_fixed_list = [i for i in mask_fixed_list if i < len(instruction_list_ori)]
            mask_fixed_list = mask_fixed_list[:mask_threshold]
            seq = ''
            for idx_list, idx_real in enumerate(mask_fixed_list):
                if idx_list != len(mask_fixed_list) - 1:
                    seq = seq + str(idx_real+1) + MASK_SEQ_INTERVAL_LIST[random_indicators['mask_seq_interval_idx']]
                else:
                    seq = seq + str(idx_real+1)
            seq = MASK_SEQ_TAG_LIST[random_indicators['mask_seq_tag_idx']].format(i=seq)
            mask_meta_instruction = mask_meta_instruction.format(seq)
            pass
        elif mask_method in ['WORD_LONG', 'WORD_SHORT']:
            mask_threshold = max(int(len(instruction_list_ori) * random_indicators['mask_ratio']),1)
            mask_meta_instruction = mask_meta_instruction.format(mask_threshold)
            pass

        instruction_all = format_overall_instruction_format.format(meta_instruction=meta_instruction,instruction_all=instruction_all)
        overall_instruction = mask_overall_instruction_format.format(meta_instruction=mask_meta_instruction,instruction_all=instruction_all)
    
    elif random_indicators['overall_indicator'] == 3:
        # format + ID ANSWER
        ID_ANSWER_METHOD_method = list(ID_ANSWER_METHOD_INSTRUCTION_PAIR_DICT.keys())[random_indicators['ID_ANSWER_method_idx']]
        if ID_ANSWER_METHOD_method == 'ANSWER_TO_INSTRUCTION_ID':
            # Create a dictionary to map responses to their instruction IDs
            response_to_id = {response: idx + 1 for idx, response in enumerate(response_list_formated)}
            # Choose a random response to ask about
            chosen_response = random.choice(response_list_formated)
            chosen_id = response_to_id[chosen_response]
            
            # Formulate the instruction and response
            instruction_template = random.choice(ID_ANSWER_INSTRUCTION)
            instruction_i_formated = instruction_template.format(chosen_response)
            response_i_formated = str(chosen_id)#f"The corresponding instruction ID of the answer is: {chosen_id}"
            
            # Generate the full instruction and response
            full_instruction = ''
            full_response = ''
            for i, data_i in enumerate(used_items_temp):
                instruction_i = data_i['instruction'] + '\n' + data_i['input'] if data_i['input'] != '' else data_i['instruction']
                response_i = data_i['output']
                instruction_digit_i = instruction_digit.format(i=i+1)
                instruction_i_formated_i = instruction_digit_i + instruction_i# + '\n' + instruction_digit_i + begin_tag + response_i + end_tag
                full_instruction += instruction_i_formated_i
                if i != len(used_items_temp) - 1:
                    full_instruction += FORMAT_INSTRUCTION_INTERVAL_TAG_LIST[random_indicators['format_instruction_interval_tag_idx']]
            # Combine the final instruction and response
            instruction_all = full_instruction + '\n\n' + instruction_i_formated
            response_all = full_response + '\n\n' + response_i_formated
            overall_instruction = format_overall_instruction_format.format(meta_instruction=meta_instruction, instruction_all=instruction_all)
    
    elif random_indicators['overall_indicator'] == 4:
        # format + ANSWER_NO_ANSWER
        mask_method = list(ANSWER_NO_ANSWER_METHOD_INSTRUCTION_PAIR_DICT.keys())[random_indicators['ANSWER_NO_ANSWER_method_idx']]
        instruction_list_formated = []
        response_list_formated = []
        for i, data_i in enumerate(used_items_temp):
            instruction_i = data_i['instruction'] + '\n' + data_i['input'] if data_i['input']!= '' else data_i['instruction']
            response_i = data_i['output']

                # Get formated instruction_i
            instruction_digit_i = instruction_digit.format(i=i+1)
            if random.random() < 0.2:  # 20% chance of not concatenating response with instruction
                instruction_i_formated = 'question ' + instruction_digit_i + ': ' + instruction_i
                response_i_formated = 'answer '+ instruction_digit_i  + ': ' + begin_tag + response_i + end_tag
            else:
                instruction_i_formated = 'question ' + instruction_digit_i + ': ' + instruction_i #+ '\n answer '+ instruction_digit_i  + ': ' + begin_tag + response_i + end_tag
                response_i_formated = ''
            instruction_list_formated.append(instruction_i_formated)
            response_list_formated.append(response_i_formated)
        masked_response_list = response_list_formated
        instruction_all, response_all = wrap_instruction_response(random_indicators, instruction_list_formated, masked_response_list)

        mask_meta_instruction = ANSWER_NO_ANSWER_METHOD_INSTRUCTION_PAIR_DICT[mask_method][random_indicators['ANSWER_NO_ANSWER_meta_instruction_idx']]
        mask_overall_instruction_format = ANSWER_NO_ANSWER_OVERALL_INSTRUCTION_LIST[random_indicators['ANSWER_NO_ANSWER_overall_instruction_idx']]
        instruction_all = format_overall_instruction_format.format(meta_instruction=meta_instruction,instruction_all=instruction_all)
        overall_instruction = mask_overall_instruction_format.format(meta_instruction=mask_meta_instruction,instruction_all=instruction_all)
    
    elif random_indicators['overall_indicator'] == 5:
        # format + QA_FEWSHOT_ANSWER
        # Randomly select one (instruction, response) pair to exclude
        exclude_index = random.randint(0, len(instruction_list_formated) - 1)
        qa_pairs = []

        # Append all but one (instruction, response) pair
        for i, (instruction_i, response_i) in enumerate(zip(instruction_list_formated, response_list_formated)):
            if i != exclude_index:
                #qa_pairs.append(f"{instruction_i}\n{response_i}")
                qa_pairs.append(f"question: {instruction_i}\n\nanswer: {response_i}")

        # Use the excluded pair as the new question and response
        new_question = instruction_list_ori[exclude_index]
        new_question_response = response_list_formated[exclude_index]

        # Create a mapping from original instructions to their formatted responses
        qa_dict = {data_i['instruction']: response_i for data_i, response_i in zip(used_items_temp, response_list_formated)}

        # Generate the prompt
        prompt_template = random.choice(QA_FEWSHOT_ANSWER)
        new_question_prompt = prompt_template.format(''.join(qa_pairs), new_question)

        # Combine all QA pairs and the new question prompt
        instruction_all = new_question_prompt
        response_all = new_question_response
        return {'instruction': instruction_all, 'output': response_all}  

     
    elif random_indicators['overall_indicator'] == 6:
    # Format + ANSWER_BEFORE_AFTER
        if len(instruction_list_ori) < 2:
        # If there are less than two instructions, simply format only
            instruction_all, response_all = wrap_instruction_response(
            random_indicators, instruction_list_formated, response_list_formated)
            overall_instruction = format_overall_instruction_format.format(
            meta_instruction=meta_instruction, instruction_all=instruction_all)
        else:
        # Randomly choose between 'ANSWER_BEFORE' and 'ANSWER_AFTER'
            answer_type = random.choice(['ANSWER_BEFORE', 'ANSWER_AFTER'])

        # Determine a valid n such that query is within bounds
            max_n_before = len(instruction_list_ori) - 1
            max_n_after = len(instruction_list_ori) - 1

            if answer_type == 'ANSWER_BEFORE':
                chosen_index = random.randint(0, len(instruction_list_ori) - 1)
                valid_n = min(chosen_index, max_n_before)
                if valid_n < 1:
                    n = 0  # Or another default behavior, such as setting to 1 if business logic allows.
                else:
                    n = random.randint(1, valid_n)
                template = ANSWER_BEFORE_AFTER_INSTRUCTION_PAIR_DICT['ANSWER_BEFORE'][random_indicators['ANSWER_BEFORE_AFTER_meta_instruction_idx']]
                response_all = response_list_formated[chosen_index - n]

            elif answer_type == 'ANSWER_AFTER':
                chosen_index = random.randint(0, len(instruction_list_ori) - 1)
                valid_n = min(len(instruction_list_ori) - chosen_index - 1, max_n_after)
                if valid_n < 1:
                    n = 0  # Or handle similarly
                else:
                    n = random.randint(1, valid_n)
                template = ANSWER_BEFORE_AFTER_INSTRUCTION_PAIR_DICT['ANSWER_AFTER'][random_indicators['ANSWER_BEFORE_AFTER_meta_instruction_idx']]
                response_all = response_list_formated[chosen_index + n]

            chosen_instruction = instruction_list_ori[chosen_index]

        # Format the overall instruction using the chosen template and n
            formatted_instruction_part = f"{chosen_instruction}"
            overall_instruction = template.format(n, formatted_instruction_part)

        # Add the instructions to the overall instruction
            instruction_all, _ = wrap_instruction_response(
            random_indicators, instruction_list_formated, response_list_formated)

            overall_instruction = format_overall_instruction_format.format(
            meta_instruction=overall_instruction, instruction_all=instruction_all)

    return {'instruction':overall_instruction, 'output':response_all}

def get_next_line_response_list(random_indicators, instruction_list_ori, response_list_formated):

    masked_response_list = []
    removed_last_sentences = []

    for response_i in response_list_formated:
            # Extract the last line of code if present
        last_line_of_code = extract_last_line_of_code(response_i)

            # Find the second last sentence boundary
        matches = list(re.finditer(r'\. ', response_i))
        if len(matches) > 1:
            second_last_sentence_boundary = matches[-2]
                # Split the response into two parts: before and after the second last sentence boundary
            masked_response_i = response_i[:second_last_sentence_boundary.end()].strip()+' {}'
            last_sentence = response_i[second_last_sentence_boundary.end():].strip()
        else:
                # If there is only one sentence boundary or none, treat the entire response as the last sentence
            masked_response_i = ''
            last_sentence = response_i.strip()

            # If there was a code block, use its last line instead of the last sentence
        if last_line_of_code:
            last_sentence = last_line_of_code
                #masked_response_i = masked_response_i

        masked_response_list.append(masked_response_i)
        removed_last_sentences.append(last_sentence)

    return masked_response_list, removed_last_sentences
    

def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--data_path", type=str, default='data/original_sft_data.json')
    parser.add_argument("--save_path", type=str, default='flora_enhanced_data.json')
    parser.add_argument("--model_name_or_path", type=str, default='meta-llama/Llama-2-7b-hf')
    parser.add_argument("--max_length", type=int, default=128000)
    parser.add_argument("--epo_num", type=int, default=1)
    parser.add_argument("--version", type=str, default='all', choices=['format', 'order', 'mask', 'both', 'all'])
    args = parser.parse_args()
    return args




import json
import random
import numpy as np
from tqdm import tqdm
from transformers import AutoTokenizer
from collections import defaultdict

def main():
    args = parse_args()
    tokenizer = AutoTokenizer.from_pretrained(args.model_name_or_path)
    #128k token max length
    w1, w2, w3, w4, w5 = 0.8281, 0.1088, 0.0275, 0.0183, 0.0173
    
    length_intervals = [
        (0,                     args.max_length // 5,               w1),
        (args.max_length // 5 + 1,   2 * args.max_length // 5,           w2),
        (2 * args.max_length // 5 + 1, 3 * args.max_length // 5,           w3),
        (3 * args.max_length // 5 + 1, 4 * args.max_length // 5,           w4),
        (4 * args.max_length // 5 + 1, args.max_length,                  w5)
    ]
    
    
    
    '''length_intervals = [
        (0, 25600, 0.8281),
        (25601, 51200, 0.1088),
        (51201, 76800, 0.0275),
        (76801, 102400, 0.0183),
        (102401, 128000, 0.0173)
    ]'''

    '''length_intervals = [
        (0, 16000, 0.8281),  # 0-16k tokens
        (16001, 32000, 0.1088),  # 16k-32k tokens
        (32001, 48000, 0.0275),  # 32k-48k tokens
        (48001, 64000, 0.0183),  # 48k-64k tokens
        (64001, 80000, 0.0173)   # 64k-80k tokens
    ]'''
    
    # 初始化分布跟踪器
    class DistributionController:
        def __init__(self, intervals):
            self.intervals = intervals
            self.target_counts = [w for _,_,w in intervals]
            self.actual_counts = [0]*len(intervals)
            self.total = 0
            
        def update(self, length):
            for idx, (min_l, max_l, _) in enumerate(self.intervals):
                if min_l <= length <= max_l:
                    self.actual_counts[idx] += 1
                    self.total += 1
                    break
                    
        def get_weights(self):
            """动态调整区间选择权重"""
            weights = []
            for target, actual in zip(self.target_counts, self.actual_counts):
                actual_ratio = actual / self.total if self.total >0 else 0
                weight = target + (target - actual_ratio)*2
                weights.append(max(0.01, weight))
            return np.array(weights)/sum(weights)

    dist_ctl = DistributionController(length_intervals)

    # 加载原始数据
    with open(args.data_path) as f:
        original_data = json.load(f)

    # 数据追踪结构
    class DataPool:
        def __init__(self, data, epochs):
            self.epochs = epochs
            self.all_items = data * epochs
            self.remaining = set(range(len(self.all_items)))
            
        def get_batch(self, target_length):
            """智能批次选择算法"""
            candidates = []
            current_length = 0
            indices = list(self.remaining)
            random.shuffle(indices)  # 随机打乱顺序，增加多样性

            for idx in indices:
                item = self.all_items[idx]
                item_len = len(tokenizer.tokenize(item['instruction'] + item['output']))
                if current_length + item_len > target_length * 1.1:  # 放宽上限至110%，避免bug
                    continue
                candidates.append( (idx, item_len) )
                current_length += item_len
                if current_length >= target_length * 0.9:  # 降低下限至90%，避免bug
                    break
            return sorted(candidates, key=lambda x: -x[1])  # 按长度降序

    data_pool = DataPool(original_data, args.epo_num)
    all_samples = []

    # 进度跟踪
    progress_bar = tqdm(total=len(data_pool.all_items), desc="Processing Data")

    # 策略使用统计
    strategy_usage = [0] * 7
    current_strategy = 0  # 新增轮询指针

    while data_pool.remaining:
        # 动态选择目标区间
        weights = dist_ctl.get_weights()
        interval_idx = np.random.choice(len(length_intervals), p=weights)
        min_l, max_l, _ = length_intervals[interval_idx]
        target_length = random.randint(min_l, max_l)

        # 获取候选批次
        batch = data_pool.get_batch(target_length)
        if not batch:
            # 处理残留数据
            idx = next(iter(data_pool.remaining))
            item = data_pool.all_items[idx]
            all_samples.append({
                "instruction": item['instruction'],
                "output": item['output'],
                "count": 1
            })
            data_pool.remaining.remove(idx)
            progress_bar.update(1)
            continue

        # 尝试所有策略
        successful_samples = []
        for strategy in range(7):
            try:
                selected = [data_pool.all_items[idx] for idx, _ in batch]
                formatted = get_formated_data_pro(
                    get_random_indicators(args, strategy), 
                    selected
                )
                total_length = len(tokenizer.tokenize(
                    formatted['instruction'] + formatted['output']
                ))
                
                # 验证长度
                if min_l <= total_length <= max_l:
                    successful_samples.append((formatted, [idx for idx, _ in batch], strategy))
            except:
                continue

        # ==== 修改开始 ==== #
        if successful_samples:
            # 按轮询顺序选择策略
            selected_sample = None
            for offset in range(7):
                try_strategy = (current_strategy + offset) % 7
                # 查找第一个符合当前策略的成功样本
                for sample in successful_samples:
                    if sample[2] == try_strategy:
                        selected_sample = sample
                        break
                if selected_sample:
                    break
            
            if selected_sample:
                formatted, used_indices, strategy = selected_sample
                # 更新指针到下一个策略
                current_strategy = (strategy + 1) % 7
            else:
                # 保底机制：如果没有策略匹配（理论上不会发生）
                formatted, used_indices, strategy = random.choice(successful_samples)

            all_samples.append({
                "instruction": formatted['instruction'],
                "output": formatted['output'],
                "count": len(used_indices),
                "strategy": strategy
            })
            data_pool.remaining -= set(used_indices)
            dist_ctl.update(len(tokenizer.tokenize(
                formatted['instruction'] + formatted['output']
            )))
            progress_bar.update(len(used_indices))
            strategy_usage[strategy] += 1
        # ==== 修改结束 ==== #

        else:
            # 处理残留数据
            idx = next(iter(data_pool.remaining))
            item = data_pool.all_items[idx]
            all_samples.append({
                "instruction": item['instruction'],
                "output": item['output'],
                "count": 1
            })
            data_pool.remaining.remove(idx)
            progress_bar.update(1)

    progress_bar.close()

    # 保存结果
    with open(args.save_path, 'w') as f:
        json.dump(all_samples, f, indent=4)

    # 验证输出
    total_original = len(original_data) * args.epo_num
    processed_count = sum(s['count'] for s in all_samples)
    print(f"\n完整性验证: {processed_count}/{total_original} ({processed_count/total_original:.1%})")

    # 分布验证
    print("\nToken长度分布:")
    interval_counts = defaultdict(int)
    for s in all_samples:
        length = len(tokenizer.tokenize(s['instruction'] + s['output']))
        for idx, (min_l, max_l, target_pct) in enumerate(length_intervals):
            if min_l <= length <= max_l:
                interval_counts[idx] += 1
                break
                
    total = len(all_samples)
    for idx, (min_l, max_l, target_pct) in enumerate(length_intervals):
        actual_pct = interval_counts[idx] / total
        print(f"区间[{min_l}-{max_l}]: 目标 {target_pct*100:.1f}% | 实际 {actual_pct*100:.1f}%")

    # 策略使用统计
    print("\n策略使用统计:")
    for i, count in enumerate(strategy_usage):
        print(f"策略 {i}: 使用次数 {count}")


if __name__ == "__main__":
    main()


