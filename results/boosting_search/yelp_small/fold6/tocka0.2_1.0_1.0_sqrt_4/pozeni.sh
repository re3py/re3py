tar -xzf py.tar.gz
python3 run_grid_search.py 60119 6 0.2 1.0 1.0 sqrt 4
tar -czf experiment.tar.gz experiment*
