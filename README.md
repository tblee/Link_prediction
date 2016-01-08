## Link Prediction with Supervised Random Walks on Github Data
This is the implementation of Supervised Random Walks from course project in Stanford's CS224W *Social and Information Network Analysis*.
The project report is [here](http://web.stanford.edu/class/cs224w/projects_2015/Applying_Link_Prediction_for_Repository_Recommendation_on_GitHub.pdf).

### Algorithm Realization
[supervisedRWfunc.py](https://github.com/tblee/CS224W_Link_prediction/blob/master/supervisedRWfunc.py) contains the functions for training a Supervised Random Walks model.
The power-iterative solvers for PageRank and the gradient of PageRank are implemented in this library as separate functions. The *L-BFGS-B* optimizer from *Scipy* is called to solve the optimization problem in training process.


