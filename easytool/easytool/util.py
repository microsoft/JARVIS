# — coding: utf-8 –
import json
import re
import os


def read_jsonline(address):
    not_mark = []
    with open(address, 'r', encoding="utf-8") as f:
        for jsonstr in f.readlines():
            jsonstr = json.loads(jsonstr)
            not_mark.append(jsonstr)
    return not_mark


def save_json(ls, address):
    json_str = json.dumps(ls, indent=4)
    with open(address, 'w', encoding='utf-8') as json_file:
        json.dump(ls, json_file, ensure_ascii=False, indent=4)


def read_json(address):
    with open(address, 'r', encoding='utf-8') as json_file:
        json_data = json.load(json_file)
    return json_data


def remove_key(item, key_to_remove):
    if isinstance(item, dict):
        if key_to_remove in item:
            del item[key_to_remove]
        for key, value in list(item.items()):  # 使用list包裹，防止字典大小改变时引发错误
            item[key] = remove_key(value, key_to_remove)
    elif isinstance(item, list):
        for index, value in enumerate(item):
            item[index] = remove_key(value, key_to_remove)
    return item


def data_clean(dic, key):
    dic = remove_key(dic, key)
    return dic


def lowercase_parameter_keys(input_dict):
    if "parameters" in input_dict and isinstance(input_dict["parameters"], dict):
        # Convert all keys in the "parameters" dictionary to uppercase
        input_dict["parameters"] = {change_name(k.lower()): v for k, v in input_dict["parameters"].items()}
    return input_dict


def build_index(base_path):
    index = {}
    for root, dirs, files in os.walk(base_path):
        for dir_name in dirs:
            if dir_name not in index:
                index[dir_name] = []
            index[dir_name].append(root)
    return index


def change_name(name):
    change_list = ["from", "class", "return", "false", "true", "id", "and", "", "ID"]
    if name in change_list:
        name = "is_" + name.lower()
    return name


def standardize(string):
    res = re.compile("[^\\u4e00-\\u9fa5^a-z^A-Z^0-9^_]")
    string = res.sub("_", string)
    string = re.sub(r"(_)\1+", "_", string).lower()
    while True:
        if len(string) == 0:
            return string
        if string[0] == "_":
            string = string[1:]
        else:
            break
    while True:
        if len(string) == 0:
            return string
        if string[-1] == "_":
            string = string[:-1]
        else:
            break
    if string[0].isdigit():
        string = "get_" + string
    return string


def get_last_processed_index(progress_file):
    """Retrieve the last processed index from the progress file."""
    if os.path.exists(progress_file):
        with open(progress_file, 'r', encoding='utf-8') as f:
            last_index = f.read().strip()
            return int(last_index) if last_index else 0
    else:
        return 0


def update_progress(progress_file, index):
    """Update the last processed index in the progress file."""
    with open(progress_file, 'w', encoding='utf-8') as f:
        f.write(str(index))


if __name__ == '__main__':
    print("util.py")
