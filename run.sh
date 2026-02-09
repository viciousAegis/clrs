# for length in 8; do
#     # python -m clrs.examples.edge_transformer_attention --test_lengths "$length" --output_dir "edge_attention_outputs-len-$length" --entropy_only --num_test 10
#     python -m clrs.examples.edge_transformer_attention --test_lengths "$length" --output_dir edge-attn-fw-len-"$length" --algorithm bridges --num_test 1
# done

## TRAINING RUNS
# algorithms=("bridges" "floyd_warshall" "mst_kruskal" "dijkstra" "articulation_points")
# algorithms=("dfs" "floyd_warshall" "mst_kruskal" "articulation_points")
algorithms=("topological_sort" "mst_prim" "lcs_length" "find_maximum_subarray_kadane")
processors=("edge_t")

for algo in "${algorithms[@]}"; do
    for processor_type in "${processors[@]}"; do
         python -m clrs.examples.run \
            --algorithms "$algo" \
            --processor_type "$processor_type" \
            --num_layers 3 \
            --nb_heads 12 \
            --learning_rate 2.5e-4 \
            --return_attention_entropy \
            --wandb_entity akshitsinha3 \
            --wandb_project GDL_CLRS30 \
            --wandb_name "$algo"-"$processor_type"-run \
            --train_steps 3000
    done
done