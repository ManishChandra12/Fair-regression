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
python3 -m src.frontier --dataset=compas
```
The final plot will be saved inside output

## LawSchool Dataset
### Preprocessing
```
python3 -m src.preprocess_lawschool
```
### Generating Accuracy vs Fairness
```
python3 -m src.frontier --dataset=lawschool
```
The final plot will be saved inside output

### Generating POF vs alpha
```
python3 -m src.pof --dataset=lawschool
```
The final plot will be saved inside output/
