tar -xzf py.tar.gz
python3 run_grid_search.py 51723 1 0.05 1.0 0.6 sqrt 8
tar -czf experiment.tar.gz experiment*
