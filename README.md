Implementation of "A Convex Framework for Fair Regression"

# Summary
A rich family of fairness metrics for regression models that take the form of a fairness regularizer is applied to the standard loss functions for linear and logistic regression. The family of fairness metrics covers the spectrum from group fairness to individual fairness along with intermediate fairness notion. By varying the weight on the fairness regularizer, the efficient frontier of the accuracy-fairness tradeoff is obtained and the severity of this trade-off is computed via a numerical quantity called Price of Fairness (PoF).

# Requirements
python_version = "3.6" <br />
[packages] <br />
numpy==1.18.2 <br />
pandas==1.0.3 <br />
cvxpy==1.0.31 <br />
sklearn==0.22.2.post1 <br />
matplotlib==3.2.1 <br />
xlrd==1.2.0 <br />

# Results


# NOTE


## Team Members
1. Sharik A (19CS60D04)
2. Manish Chandra (19CS60A01)
3. Anju Punuru (19CS60R07)
4. Kunal Devanand Zodape (19CS60R13)
5. Anirban Saha (19CS60R50)
6. Hasmita Kurre (19CS60R67)


## Project Setup
1. Clone the repo
2. Install pipenv
```
pip install pipenv
```
3. cd to the project directory
4. Create the virtual environment
```
pipenv install --skip-lock
```
5. Activate the virtual environment
```
pipenv shell
```

### Preprocessing
```
python3 -m src.preprocess_compas
```
Replace preprocess_compas with preprocess_adult or preprocess_default for Adult and Default datasets respectively.

### Generating Accuracy vs Fairness
```
python3 -m src.frontier --dataset=compas --proc=<number of cores to use>
```
Replace compas with adult, lawschool or default for Adult, Law School and Default datasets respectively.

### Generating Price of Fairness Bar Graph
```
python3 -m src.pof --dataset=compas --proc=<number of cores to use>
```
Replace compas with adult, lawschool or default for Adult, Law School and Default datasets respectively.

The final plots will be saved inside output/
