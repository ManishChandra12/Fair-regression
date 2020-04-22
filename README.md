# Fair-regression

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

## COMPAS Dataset
### Preprocessing
```
python3 -m src.preprocess_compas
```
### Generating Accuracy vs Fairness
```
python3 -m src.frontier --dataset=compas --proc=<number of cores to use>
```
### Generating Price of Fairness Bar Graph
```
python3 -m src.pof --dataset=compas --proc=<number of cores to use>
```
The final plots will be saved inside output/

## Law School Dataset
### Generating Accuracy vs Fairness
```
python3 -m src.frontier --dataset=lawschool --proc=<number of cores to use>
```
### Generating Price of Fairness Bar Graph
```
python3 -m src.pof --dataset=lawschool --proc=<number of cores to use>
```
The final plots will be saved inside output/

## Default Dataset
### Preprocessing
```
python3 -m src.preprocess_default
```
### Generating Accuracy vs Fairness
```
python3 -m src.frontier --dataset=default --proc=<number of cores to use>
```
### Generating Price of Fairness Bar Graph
```
python3 -m src.pof --dataset=default --proc=<number of cores to use>
```
The final plots will be saved inside output/

## Adult Dataset
### Preprocessing
```
python3 -m src.preprocess_adult
```
### Generating Accuracy vs Fairness
```
python3 -m src.frontier --dataset=adult --proc=<number of cores to use>
```
### Generating Price of Fairness Bar Graph
```
python3 -m src.pof --dataset=adult --proc=<number of cores to use>
```
The final plots will be saved inside output/
