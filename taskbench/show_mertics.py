import argparse
import json

parser = argparse.ArgumentParser()
parser.add_argument("--data_dir", type=str, default="data_multimedia")
parser.add_argument("--result_path", type=str, default="metrics_use_demos_2_reformat_by_self/MiniMax-Text-01.json")
args = parser.parse_args()

result = json.load(open(f"{args.data_dir}/{args.result_path}", "r"))
result_overall = result["overall_overall"]
result_single = result["single_overall"]
result_chain = result["chain_overall"]
result_dag = result["dag_overall"]

r1 = result_overall["step_rouge1"]
r2 = result_overall["step_rouge2"]
rl = result_overall["step_rougeL"]
rlsum = result_overall["step_rougeLsum"]
# bertscore_p = result_overall["step_bertscore_precision"]
# bertscore_r = result_overall["step_bertscore_recall"]
# bertscore_f1 = result_overall["step_bertscore_f1"]
overall_n_f1 = result_overall["node_micro_f1_no_matching"]
overall_e_f1 = result_overall["link_binary_f1"]
overall_NED = result_overall["edit_distance"]
overall_t_f1 = result_overall["argument_task_argname_binary_f1_no_matching"]
overall_v_f1 = result_overall["argument_task_argname_value_binary_f1_no_matching"]

single_n_f1 = result_single["node_micro_f1_no_matching"]
single_t_f1 = result_single["argument_task_argname_binary_f1_no_matching"]
single_v_f1 = result_single["argument_task_argname_value_binary_f1_no_matching"]

chain_n_f1 = result_chain["node_micro_f1_no_matching"]
chain_e_f1 = result_chain["link_binary_f1"]
chain_NED = result_chain["edit_distance"]
chain_t_f1 = result_chain["argument_task_argname_binary_f1_no_matching"]
chain_v_f1 = result_chain["argument_task_argname_value_binary_f1_no_matching"]

dag_n_f1 = result_dag["node_micro_f1_no_matching"]
dag_e_f1 = result_dag["link_binary_f1"]
dag_NED = result_dag["edit_distance"]
dag_t_f1 = result_dag["argument_task_argname_binary_f1_no_matching"]
dag_v_f1 = result_dag["argument_task_argname_value_binary_f1_no_matching"]

# 创建 Markdown 表格
markdown_table = """
# Task Decomposition
<table>
    <tr>
        <td colspan="3">{data_dir}</td>
    </tr>
    <tr>
        <td>R1</td>
        <td>R2</td>
        <td>RL</td>
    </tr>
    <tr>
        <td>{r1}</td>
        <td>{r2}</td>
        <td>{rl}</td>
    </tr>
</table>

# Tool Selection
<table>
    <tr>
        <td>Node</td>
        <td colspan="3">Chain</td>
        <td colspan="3">DAG</td>
        <td colspan="3">Overall</td>
    </tr>
    <tr>
        <td>n-F1</td>
        <td>n-F1</td>
        <td>e-F1</td>
        <td>NED</td>
        <td>n-F1</td>
        <td>e-F1</td>
        <td>NED</td>
        <td>n-F1</td>
        <td>e-F1</td>
        <td>NED</td>
    </tr>
    <tr>
        <td>{single_n_f1}</td>
        <td>{chain_n_f1}</td>
        <td>{chain_e_f1}</td>
        <td>{chain_NED}</td>
        <td>{dag_n_f1}</td>
        <td>{dag_e_f1}</td>
        <td>{dag_NED}</td>
        <td>{overall_n_f1}</td>
        <td>{overall_e_f1}</td>
        <td>{overall_NED}</td>
    </tr>
</table>

# Tool Parameter Prediction
<table>
    <tr>
        <td colspan="2">Node</td>
        <td colspan="2">Chain</td>
        <td colspan="2">DAG</td>
        <td colspan="2">Overall</td>
    </tr>
    <tr>
        <td>t-F1</td>
        <td>v-F1</td>
        <td>t-F1</td>
        <td>v-F1</td>
        <td>t-F1</td>
        <td>v-F1</td>
        <td>t-F1</td>
        <td>v-F1</td>
    </tr>
    <tr>
        <td>{single_t_f1}</td>
        <td>{single_v_f1}</td>
        <td>{chain_t_f1}</td>
        <td>{chain_v_f1}</td>
        <td>{dag_t_f1}</td>
        <td>{dag_v_f1}</td>
        <td>{overall_t_f1}</td>
        <td>{overall_v_f1}</td>
    </tr>
</table>
""".format(
    data_dir=args.data_dir,
    r1=r1, r2=r2, rl=rl,
    single_n_f1=single_n_f1, chain_n_f1=chain_n_f1, dag_n_f1=dag_n_f1, overall_n_f1=overall_n_f1,
    chain_e_f1=chain_e_f1, dag_e_f1=dag_e_f1, overall_e_f1=overall_e_f1,
    chain_NED=chain_NED, dag_NED=dag_NED, overall_NED=overall_NED,
    single_t_f1=single_t_f1, chain_t_f1=chain_t_f1, dag_t_f1=dag_t_f1, overall_t_f1=overall_t_f1,
    single_v_f1=single_v_f1, chain_v_f1=chain_v_f1, dag_v_f1=dag_v_f1, overall_v_f1=overall_v_f1
)

# 保存 Markdown 表格到文件
with open(f"{args.data_dir}/{args.result_path.replace('.json', '_metrics_table.md')}", "w") as file:
    file.write(markdown_table)








