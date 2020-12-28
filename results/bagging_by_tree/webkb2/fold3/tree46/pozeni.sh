tar -xzf py.tar.gz
python3 run_experiment_by_tree.py -port None -fold 3 -model DT -rs 130083 -mret 1.0 -at 3
tar -czf experiment.tar.gz experiment*

