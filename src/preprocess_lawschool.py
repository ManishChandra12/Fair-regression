import pandas as pd
import os
from CONSTANTS import RAW_DATA_DIR, PROCESSED_DATA_DIR

def main():
	# Law school Dataset contains records for law students who took the Bar exam and whether passed it or not
	df = pd.read_csv(os.path.join(RAW_DATA_DIR, 'Law School/lawschool.csv'))
	
	# Gender is the sensitive attribute getting it's unique values
	gender_uniq = df['gender'].unique()

	# Convert categorical variables into dummy/indicator variables
	gender = pd.get_dummies(df['gender'])

	# Drop the categorical features and append the corresponding indicator variables
	df.drop(['gender'], axis=1, inplace=True)
	df = pd.concat([df, gender], axis=1)

	# Does the dataframe have any missing value
	assert df.isnull().values.any() == False

	#df.rename(columns={'0': 'Female','1': 'Male'}, inplace=True)
	df.columns = ['cluster', 'lsat', 'ugpa','zfygpa','zgpa','bar1','fulltime','fam_inc','age','race1','race2','race3','race4','race5','race6','race7','race8','female','male']

	# Dump into csv
	if not os.path.isdir(os.path.join(PROCESSED_DATA_DIR, 'LAWSCHOOL')):
		os.mkdir(os.path.join(PROCESSED_DATA_DIR, 'LAWSCHOOL'))
	df.to_csv(os.path.join(PROCESSED_DATA_DIR,'LAWSCHOOL/lawschool_processed.csv'), index=False)

	# Create csvs for separate models setting
	for sensitive_attr_val in gender_uniq:
		if (sensitive_attr_val == 0):
			sensitive_attr = 'female'
		else:
			sensitive_attr = 'male'
		split_df = df[df[sensitive_attr] == 1]
		split_df = split_df.drop(sensitive_attr, axis=1)
		split_df.to_csv(os.path.join(PROCESSED_DATA_DIR,'LAWSCHOOL/lawschool_' + sensitive_attr + '.csv'), index=False)

if __name__ == '__main__':
	main()
