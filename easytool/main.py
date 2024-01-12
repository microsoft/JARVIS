# — coding: utf-8 –
import openai
import json
import argparse
import os
from tqdm import tqdm
from easytool import funcQA, restbench, toolbench_retrieve, toolbench
from easytool.util import *
openai.api_key = os.environ["OPENAI_API_KEY"]
        
if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--model_name', type=str, default='gpt-3.5-turbo')
    parser.add_argument('--task', type=str, default='funcqa_mh', help='funcqa, toolbench_retrieve, toolbench, restbench')
    parser.add_argument('--data_type', type=str, default='G3', help='G2 or G3 or funcqa_mh or funcqa_oh')
    parser.add_argument('--tool_root_dir', type=str, default='.toolenv/tools/')
    parser.add_argument('--retrieval_num', type=int, default=5)
    
    args = parser.parse_args()
    
    if args.task == 'funcqa':
        dataset = read_json('data_funcqa/tool_instruction/functions_data.json')
        Tool_dic = read_jsonline('data_funcqa/tool_instruction/tool_dic.jsonl')
        test_data = read_json(f"data_funcqa/test_data/{args.data_type}.json")
        progress_file = f"FuncQA_{args.data_type}_{args.model_name}_Easytool.txt"
    
        
    elif 'toolbench' in args.task:
        base_path = args.tool_root_dir
        index = build_index(base_path)
        dataset = read_json('data_toolbench/tool_instruction/toolbench_tool_instruction.json')
        if args.data_type == 'G2':
            test_data = read_json(f'''data_toolbench/test_data/{args.data_type}_category.json''')
        elif args.data_type == 'G3':
            test_data = read_json(f'''data_toolbench/test_data/{args.data_type}_instruction.json''')
        progress_file = f'''{args.data_type}_{args.model_name}_Easytool.txt'''
        
    
    elif args.task == 'restbench':
        Tool_dic = read_json('data_restbench/tool_instruction/tmdb_tool.json')
        dic_tool = {}
        for data in Tool_dic:
            dic_tool[data['ID']] = data
        test_data = read_json('data_restbench/test_data/tmdb.json')
        progress_file = f"restbench_{args.model_name}_Easytool.txt"

    else:
        print("Wrong task name")
        exit()  
        
    start_index = get_last_processed_index(progress_file)
    total_files = len(test_data)
    retrieval_num = args.retrieval_num
    ind = start_index
    model_name = args.model_name
    
    print("-------Start Execution-------")
    if args.data_type == 'funcqa_mh':
        funcQA.task_execution_mh(args.data_type, start_index, total_files, 
                                        retrieval_num, ind, model_name, dataset, 
                                        Tool_dic, test_data, progress_file)
    elif args.data_type == 'funcqa_oh':
        funcQA.task_execution_oh(args.data_type, start_index, total_files, 
                                        retrieval_num, ind, model_name, dataset, 
                                        Tool_dic, test_data, progress_file)
        
        
    elif args.task == 'toolbench_retrieve':
        toolbench_retrieve.task_execution(args.data_type,
            base_path, index, dataset, test_data, progress_file, 
            start_index, total_files, retrieval_num, ind, model_name)

        
    
    elif args.task == 'toolbench':
        toolbench.task_execution(args.data_type,
            base_path, index, dataset, test_data, progress_file, 
            start_index, total_files, retrieval_num, ind, model_name)

        
    
    elif args.task == 'restbench':
        restbench.task_execution(
            Tool_dic, dic_tool, test_data, progress_file, 
            start_index, total_files, retrieval_num, ind, model_name)

    
    else:
        print("Wrong task name")
        exit()
