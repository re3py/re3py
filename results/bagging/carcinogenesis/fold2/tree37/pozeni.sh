tar -xzf py.tar.gz
python3 run_experiment_by_tree.py -port None -fold 2 -model DT -rs 149493 -mret 1.0 -at 2
tar -czf experiment.tar.gz experiment*

