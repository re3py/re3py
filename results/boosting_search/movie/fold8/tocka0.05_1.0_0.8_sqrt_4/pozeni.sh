tar -xzf py.tar.gz
python3 run_grid_search.py 38423 8 0.05 1.0 0.8 sqrt 4
tar -czf experiment.tar.gz experiment*
