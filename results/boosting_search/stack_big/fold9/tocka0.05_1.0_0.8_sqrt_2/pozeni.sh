tar -xzf py.tar.gz
python3 run_grid_search.py 50517 9 0.05 1.0 0.8 sqrt 2
tar -czf experiment.tar.gz experiment*
