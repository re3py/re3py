tar -xzf py.tar.gz
python3 run_grid_search.py 60359 6 0.05 1.0 0.6 sqrt 4
tar -czf experiment.tar.gz experiment*
