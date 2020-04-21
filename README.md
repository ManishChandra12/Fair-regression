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

## Default Dataset
### Preprocessing
```
python3 -m src.preprocess_default
```
### Generating Accuracy vs Fairness
```
python3 -m src.frontier --dataset=default
```
The final plot will be saved inside output/

### Generating POF
```
python3 -m src.pof --dataset=default
```
The final plot will be saved inside output/

