tar -xzf py.tar.gz
python3 run_grid_search.py 50901 0 0.2 1.0 1.0 sqrt 2
tar -czf experiment.tar.gz experiment*
