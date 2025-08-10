python inference.py \
    --llm MiniMax-Text-01 \
    --data_dir data_dailylifeapis \
    --temperature 0.001 \
    --top_p 0.1 \
    --api_addr localhost \
    --api_port 4000 \
    --api_base https://api.minimax.chat/v1 \
    --api_key xxx \
    --multiworker 5 \
    --use_demos 0 \
    --reformat true \
    --reformat_by self \
    --log_first_detail true \
    --use_demos 2 \
    --dependency_type temporal \
    --tag true


python evaluate.py \
    --data_dir data_multimedia \
    --prediction_dir predictions_use_demos_2_reformat_by_self \
    --file_name MiniMax-Text-01 \
    --splits all \
    --n_tools all \
    --mode add \
    --dependency_type resource \
    -m all


python show_mertics.py \
    --data_dir data_multimedia \
    --result_path metrics_use_demos_2_reformat_by_self/MiniMax-Text-01_temperature_0.001_topp_0.1_dependency_resource.json


python generate_graph.py \
    --tool_desc data_selfbuild/tool_desc.json \
    --dependency_type resource \
    --data_dir data_selfbuild


python visualize_graph.py \
    --data_dir data_selfbuild


python data_engine.py \
    --graph_desc data_selfbuild/graph_desc.json \
    --tool_desc data_selfbuild/tool_desc.json \
    --llm Minimax-Text-01 \
    --temperature 0.8 \
    --data_dir data_selfbuild \
    --top_p 1.0 \
    --dependency_type resource \
    --api_base https://api.minimax.chat/v1 \
    --save_figure false \
    --api_addr localhost \
    --api_port 4000 \
    --api_key xxx \
    --check false \
    --use_async true \
    --number_of_samples 5 \
    --multiworker 100