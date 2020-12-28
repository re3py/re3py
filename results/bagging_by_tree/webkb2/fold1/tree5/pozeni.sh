tar -xzf py.tar.gz
python3 run_experiment_by_tree.py -port None -fold 1 -model DT -rs 128673 -mret 1.0 -at 3
tar -czf experiment.tar.gz experiment*

