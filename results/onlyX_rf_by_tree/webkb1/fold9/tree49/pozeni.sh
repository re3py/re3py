tar -xzf py.tar.gz
python3 run_experiment_by_tree.py -port None -fold 9 -model DT -rs 128113 -mret sqrt -at 2
tar -czf experiment.tar.gz experiment*

