algorithms=("dfs")
# algorithms=("dfs" "floyd_warshall" "mst_kruskal" "articulation_points")
# algorithms=("topological_sort" "mst_prim" "lcs_length" "find_maximum_subarray_kadane")
processors=("edge_t" "edge_t_scaled")
lengths=(32)
# lengths=(8)

for algo in "${algorithms[@]}"; do
    for processor_type in "${processors[@]}"; do
        for length in "${lengths[@]}"; do
            python -m clrs.examples.run \
                --algorithms "$algo" \
                --processor_type "$processor_type" \
                --num_layers 3 \
                --nb_heads 12 \
                --learning_rate 2.5e-4 \
                --test_lengths "$length" \
                --return_attention_entropy \
                --test_only \
                --attention_logit_scale 5.0 \
                # --wandb_entity akshitsinha3 \
                # --wandb_project GDL_CLRS30 \
                # --wandb_name "$algo"-"$processor_type"-"$length"-test-run
        done
    done
done