tar -xzf py.tar.gz
python3 run_experiment_by_tree.py -port None -fold 5 -model DT -rs 130953 -mret 1.0 -at 3
tar -czf experiment.tar.gz experiment*

