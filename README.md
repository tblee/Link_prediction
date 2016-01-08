## Link Prediction with Supervised Random Walks on Github Data
This is the implementation of Supervised Random Walks from course project in Stanford's CS224W *Social and Information Network Analysis*.
The project report is [here](http://web.stanford.edu/class/cs224w/projects_2015/Applying_Link_Prediction_for_Repository_Recommendation_on_GitHub.pdf).

### Supervised Random Walks Realization
[supervisedRWfunc.py](https://github.com/tblee/CS224W_Link_prediction/blob/master/supervisedRWfunc.py) contains the functions for training a Supervised Random Walks model.
The power-iterative solvers for PageRank and the gradient of PageRank are implemented in this library as separate functions. The *L-BFGS-B* optimizer from *Scipy* is called to solve the optimization problem in training process.

### Link Prediction of Repository Graph
[gitRepoTrain.py](https://github.com/tblee/CS224W_Link_prediction/blob/master/gitRepoTrain.py) implements link prediction on repository graph by training model with Supervised Random Walks.
The prediction performance comparing to Unweighted random walk is written into a *.json* file.

### Repository Recommendation to Users
[userRepoRecom.py](https://github.com/tblee/CS224W_Link_prediction/blob/master/userRepoRecom.py) recommends repository to users by Personalized PageRank.
The comparison between Supervised Random Walks based and Unweighted random walk based repository recommendation is plotted in the end.



