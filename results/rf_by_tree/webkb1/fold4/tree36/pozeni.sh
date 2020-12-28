tar -xzf py.tar.gz
python3 run_experiment_by_tree.py -port None -fold 4 -model DT -rs 125483 -mret sqrt -at 2
tar -czf experiment.tar.gz experiment*

