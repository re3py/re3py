tar -xzf py.tar.gz
python3 run_experiment_by_tree.py -port None -fold 6 -model DT -rs 131213 -mret 1.0 -at 3
tar -czf experiment.tar.gz experiment*

