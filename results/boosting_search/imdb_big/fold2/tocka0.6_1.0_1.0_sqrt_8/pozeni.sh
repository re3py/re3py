tar -xzf py.tar.gz
python3 run_grid_search.py 40251 2 0.6 1.0 1.0 sqrt 8
tar -czf experiment.tar.gz experiment*
