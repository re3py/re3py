tar -xzf py.tar.gz
python3 run_experiment_by_tree.py -port None -fold 1 -model DT -rs 143973 -mret 1.0 -at 2
tar -czf experiment.tar.gz experiment*

