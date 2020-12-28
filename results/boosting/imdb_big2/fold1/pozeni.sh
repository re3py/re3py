echo hip
tar -xzf py.tar.gz
echo hop
python3 run_experiment.py 30023 1 GB 0.05 1.0 0.6 1.0 8
tar -czf experiment.tar.gz experiment*

