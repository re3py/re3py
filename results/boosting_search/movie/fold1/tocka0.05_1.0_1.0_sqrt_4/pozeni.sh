tar -xzf py.tar.gz
python3 run_grid_search.py 34343 1 0.05 1.0 1.0 sqrt 4
tar -czf experiment.tar.gz experiment*
