# NN_OptimizedWithPSO

- The main.py file can run any function. Select it commenting lines 83-88.
- The files [1in_cubic.py, 1in_linear.py, 1in_sine.py, 1in_tanh.py, 2in_complex.py, 2in_xor.py] each run the respective function with the hyperparameters that gave the best results during testing.
- The PSO algorithm is run inside the draw_graph function for 750 iterations.
- In every iteration, draw_graph plots 2 graphs. One with the desired output function and the best solution ever found by the swarm, and another showing the evolution of the MSE.
- The test.py file is the script used to test all the hyperparameters
- The graph_painter.py file is the script used to create [error_layers.png, error_pbestinfluence.png, error_popsize.png]