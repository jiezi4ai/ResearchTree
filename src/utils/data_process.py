import copy 
import json
import string
from typing import List, Dict, Optional
from thefuzz import fuzz # pip install thefuzz  https://github.com/seatgeek/thefuzz


# string related opeartions
def remove_non_text_chars(text, with_digits: Optional[bool]=True):
    """remove non text chars
    """
    valid_chars = string.ascii_letters
    if with_digits == True:
        valid_chars += string.digits  # 包含所有字母和数字的字符串
    cleaned_text = ''
    for char in text:
        if char in valid_chars:
            cleaned_text += char
    return cleaned_text

def text_match(text_a, text_b, with_digits: Optional[bool]=True):
    """"fuzzy match between text_a and text_b"""
    text_a = remove_non_text_chars(text_a, with_digits).lower()
    text_b = remove_non_text_chars(text_b, with_digits).lower()
    return fuzz.ratio(text_a, text_b)


def text_patial_match(shorter_text, longer_text, with_digits: Optional[bool]=True):
    """"partial fuzzy match between text_a and text_b"""
    shorter_text = remove_non_text_chars(shorter_text, with_digits).lower()
    longer_text = remove_non_text_chars(longer_text, with_digits).lower()
    return fuzz.partial_ratio(shorter_text, longer_text)


# list related operations
def remove_kth_element(original_list, k):
    """删除list中第k个元素 (不改变原list的值，仅返回新list)"""
    if k <= 0 or k > len(original_list):
        return list(original_list)  # 返回原list的副本，不改变原list
    else:
        new_list = list(original_list) # 创建原list的副本
        new_list.pop(k - 1) # 删除索引为 k-1 的元素 (因为list索引是 0-based)
        return new_list


# dict related operations
def rename_key_in_dict(input_dict, key_mapping):
    """rename keys in dict
    Args:
        key_mapping: Mapping of old keys to new keys, in format like: {"old_name_1": "new_name_1", "old_name_2": "new_name_2", ...}  
    """
    return {key_mapping.get(k, k): v for k, v in input_dict.items()}

def move_key_to_first(input_dict, key_to_move):
    """move a specific key to the first"""
    if key_to_move not in input_dict:
        return input_dict  # 如果键不存在，则直接返回原字典

    value = input_dict[key_to_move]
    new_dict = {key_to_move: value} # 创建新字典，首先插入要移动的键
    for k, v in input_dict.items():
        if k != key_to_move:
            new_dict[k] = v
    return new_dict

def filter_and_reorder_dict(input_dict, keys_to_keep):
    """filter and re-order keys of dict"""
    return {key: input_dict[key] for key in keys_to_keep if key in input_dict}

def remove_key_values(input_dict, keys_to_delete):
    """delete key-value in dict"""
    opt_dct = copy.deepcopy(input_dict)
    for key in keys_to_delete:
        if key in opt_dct:  # 检查键是否存在，避免 KeyError
            del opt_dct[key]
    return opt_dct # 为了方便链式调用，返回修改后的字典

def convert_dict_values_to_json(dict_data):
    """检查字典的值，如果值是字典类型，则将其转换为 JSON 字符串。
    Args:
        dict_data (dict): 输入字典。
    Returns:
        dict: 值被转换后的字典。
    """
    modified_dict = {}
    for key, value in dict_data.items():
        if isinstance(value, dict):
            modified_dict[key] = json.dumps(value, ensure_ascii=False)
        else:
            modified_dict[key] = value
    return modified_dict