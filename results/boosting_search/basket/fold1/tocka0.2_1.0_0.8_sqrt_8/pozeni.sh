tar -xzf py.tar.gz
python3 run_grid_search.py 51531 1 0.2 1.0 0.8 sqrt 8
tar -czf experiment.tar.gz experiment*
