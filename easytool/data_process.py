import os
import gdown
import shutil
import json
from zipfile import ZipFile

urls = {
    "funcqa": "https://drive.google.com/uc?id=13Sj7uIsyqWXoTh1ejWUviTzeQSES2Omd",
    "restbench": "https://raw.githubusercontent.com/Yifan-Song793/RestGPT/main/datasets/tmdb.json",
    "toolbench": "https://drive.google.com/uc?id=1XFjDxVZdUY7TXYF2yvzx3pJlS2fy78jk",
}


def read_jsonline(address):
    not_mark = []
    with open(address, 'r', encoding="utf-8") as f:
        for jsonstr in f.readlines():
            jsonstr = json.loads(jsonstr)
            not_mark.append(jsonstr)
    return not_mark


def read_json(address):
    with open(address, 'r', encoding='utf-8') as json_file:
        json_data = json.load(json_file)
    return json_data


def toolbench_process(data_file, dataset):
    ls = read_json(data_file)
    all_data = read_json(f"{dataset}/tool_instruction/toolbench_tool_instruction.json")
    all_dic = {}
    for ID in all_data.keys():
        all_dic[all_data[ID]["tool_name"]] = all_data[ID]

    not_in = []
    for data in ls:
        Tool_dic = []
        data_dic = {}
        already = []
        for tool in data['api_list']:
            if tool['tool_name'] in all_dic:
                if all_dic[tool['tool_name']]["ID"] not in already:
                    already.append(all_dic[tool['tool_name']]["ID"])
                    Tool_dic.append({"ID": all_dic[tool['tool_name']]["ID"],
                                     "Description": all_dic[tool['tool_name']]["tool_description"], })
        data["Tool_dic"] = Tool_dic

    json_str = json.dumps(ls, indent=4)
    with open(data_file, 'w', encoding='utf-8') as json_file:
        json.dump(ls, json_file, ensure_ascii=False, indent=4)


def main():
    curr_dir = os.path.dirname(__file__)

    for dataset in [
        "funcqa",
        "restbench",
        "toolbench"
    ]:
        dataset_path = os.path.join(curr_dir, "data_{}".format(dataset), "test_data")

        if not os.path.exists(dataset_path):
            os.makedirs(dataset_path)

        if dataset == "funcqa":
            print("Processing FuncQA dataset ...\n")
            # Processing FuncQA dataset ...
            temp_file = os.path.join(dataset_path, "data_toolkengpt_0918.zip")
            gdown.download(urls[dataset], temp_file, quiet=False)
            zf = ZipFile(temp_file, 'r')
            zf.extract("data/funcqa/funcqa_oh.json", ".")
            zf.extract("data/funcqa/funcqa_mh.json", ".")
            os.rename("data/funcqa/funcqa_oh.json", "{}/funcqa_oh.json".format(dataset_path))
            os.rename("data/funcqa/funcqa_mh.json", "{}/funcqa_mh.json".format(dataset_path))
            os.remove(temp_file)
            shutil.rmtree("data")

            print("FuncQA dataset Done!\n")

        if dataset == "restbench":
            print("Processing RestBench dataset ... \n")
            # Processing RestBench Dataset
            os.system("wget -P {} -c {}".format(dataset_path, urls[dataset]))

            print("RestBench dataset Done!\n")

        if dataset == "toolbench":
            print("Processing ToolBench dataset ... \n")
            # Processing ToolBench Dataset
            temp_file = os.path.join(dataset_path, "data.zip")
            gdown.download(urls[dataset], temp_file, quiet=False)
            zf = ZipFile(temp_file, 'r')
            zf.extract("data/test_instruction/G2_category.json", ".")
            zf.extract("data/test_instruction/G3_instruction.json", ".")

            os.rename("data/test_instruction/G2_category.json", "{}/G2_category.json".format(dataset_path))
            os.rename("data/test_instruction/G3_instruction.json", "{}/G3_instruction.json".format(dataset_path))
            toolbench_process("{}/G2_category.json".format(dataset_path), "data_{}".format(dataset))
            toolbench_process("{}/G3_instruction.json".format(dataset_path), "data_{}".format(dataset))
            os.remove(temp_file)
            shutil.rmtree("data")

            print("Toolbench dataset Done!\n")


if __name__ == '__main__':
    main()
