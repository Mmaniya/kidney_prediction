from keras.models import Sequential, load_model
import numpy as np
import pandas as pd

def test(self, alpha=0.01, iters=1000, fit_offset=True, verbose=False):
        self.alpha = alpha
        self.iters = iters
        self.fit_offset = fit_offset
        self.verbose = verbose

def __add_intercept(self, X):
    intercept = np.ones((X.shape[0], 1))
    return np.concatenate((intercept, X), axis=1)

def __sigmoid(self, z):
    return 1 / (1 + np.exp(-z))

def __loss(self, h, y):
    return (-y * np.log(h) - (1 - y) * np.log(1 - h)).mean()

def fit(self, X, y):
    if self.fit_offset:
         X = self.__add_intercept(X)

         self.theta = np.zeros(X.shape[1])

    for i in range(self.iters):
         z = np.dot(X, self.theta)
         h = self.__sigmoid(z)
         gradient = np.dot(X.T, (h - y)) / y.size
         self.theta -= self.alpha * gradient

         z = np.dot(X, self.theta)
         h = self.__sigmoid(z)
         loss = self.__loss(h, y)

         if(self.verbose ==True and i % 100 == 0):
            print(f'loss: {loss} \t')

def get_predicted_prob(self, X):
  if self.fit_offset:
   X = self.__add_intercept(X)

   return self.__sigmoid(np.dot(X, self.theta))

def get_predicted_class(self, X):
    return self.get_predicted_prob(X).round()

def importdata():
    balance_data = pd.read_csv('Preprocessed.csv', sep= ',', header = 0)

    # Printing the dataswet shape
    print ("Dataset Lenght: ", len(balance_data))
    print ("Dataset Shape: ", balance_data.shape)

    # Printing the dataset obseravtions
    return balance_data

# Function to split the dataset
def splitdataset(balance_data):

    # Seperating the target variable
    X = balance_data.values[:, 0:24]
    Y = balance_data.values[:, -1]

    #print(X)
    #print(Y)

    # Spliting the dataset into train and test
    X_train, X_test, y_train, y_test = train_test_split(
    X, Y, test_size = 0.3, random_state = 100)

    return X, Y, X_train, X_test, y_train, y_test

# Function to perform training with giniIndex.
def train_using_gini(X_train, X_test, y_train):

    # Creating the classifier object
    clf_gini = DecisionTreeClassifier(criterion = "gini",
            random_state = 100,max_depth=3, min_samples_leaf=5)

    # Performing training
    clf_gini.fit(X_train, y_train)
    return clf_gini

# Function to perform training with entropy.
def tarin_using_entropy(X_train, X_test, y_train):

    # Decision tree with entropy
    clf_entropy = DecisionTreeClassifier(criterion = "entropy", random_state = 100,
            max_depth = 3, min_samples_leaf = 5)

    clf_entropy.fit(X_train, y_train)
    return clf_entropy


def prediction(X_test, clf_object):

    # Predicton on test with giniIndex
    y_pred = clf_object.predict(X_test)
    return y_pred

def cal_accuracy(y_test, y_pred):

    print("Confusion Matrix: \n",
    confusion_matrix(y_test,y_pred))

    print ("Accuracy : ",
    accuracy_score(y_test,y_pred)*100)

    print("Report : \n",
    classification_report(y_test, y_pred))


def run(out_dir, config_fname, data_paths_fname, stats_list_fname, split_fname=None, check_if_file_exists=False, verbose=True): 

	data_paths = util.read_yaml(data_paths_fname)
	config = util.read_yaml(config_fname)

	stats_key = config['stats_key']
	outcome_stat_name = config['outcome_stat_name']
	cohort_stat_name = config.get('cohort_stat_name', None)
	lab_lower_bound = config.get('lab_lower_bound', None)
	lab_upper_bound = config.get('lab_upper_bound', None)
	gap_days = config.get('gap_days', None)
	training_window_days = config['training_window_days']
	buffer_window_days = config['buffer_window_days']
	outcome_window_days = config['outcome_window_days']
	time_period_days = config['time_period_days']
	time_scale_days = config['time_scale_days']
	use_just_labs = config['use_just_labs']
	feature_loincs_fname = config['feature_loincs_fname']
	add_age_sex = config['add_age_sex']
	calc_gfr = config['calc_gfr']
	regularizations = config.get('regularizations', [1])
	lin_n_cv_iters = config.get('lin_n_cv_iters', -1)
	n_cv_iters = config.get('n_cv_iters', -1)
	progression = config['progression']
	progression_lab_lower_bound = config.get('progression_lab_lower_bound', None)
	progression_lab_upper_bound = config.get('progression_lab_upper_bound', None)
	progression_gap_days = config.get('progression_gap_days', None)
	progression_stages = config.get('progression_stages', None)
	progression_init_stages = config.get('progression_init_stages', None)
	evaluate_nn = config.get('evaluate_nn', True)

	outcome_fname = out_dir + stats_key + '_' + outcome_stat_name + '.txt'
	if cohort_stat_name is None:
		cohort_fname = data_paths['demographics_fname']	
	else:
		cohort_fname = out_dir + stats_key + '_' + cohort_stat_name + '.txt'
	gfr_loincs = util.read_list_files('data/gfr_loincs.txt')
	training_data_fname = out_dir + stats_key + '_training_data.txt'

	feature_loincs = util.read_list_files(feature_loincs_fname)
	if use_just_labs == False:
		feature_diseases = [[icd9] for icd9 in util.read_list_files('data/kidney_disease_mi_icd9s.txt')]
		feature_drugs = [util.read_list_files('data/drug_class_'+dc.lower().replace('-','_').replace(',','_').replace(' ','_')+'_ndcs.txt') for dc in util.read_list_files('data/kidney_disease_drug_classes.txt')]
	else: 
		feature_diseases = []	
		feature_drugs = []

	n_labs = len(feature_loincs)

	if add_age_sex:
		age_index = len(feature_loincs) + len(feature_diseases) + len(feature_drugs)
		gender_index = len(feature_loincs) + len(feature_diseases) + len(feature_drugs) + 1
	else:
		age_index = None
		gender_index = None

	features_fname = out_dir + stats_key + '_features.h5'
	features_split_fname = out_dir + stats_key + '_features_split.h5'
	predict_fname = out_dir + stats_key + '_prediction_results.yaml'
	if evaluate_nn:
		nn_predict_fname = out_dir + stats_key + '_nn_prediction_results.yaml'
	else:
		nn_predict_fname = None

	if verbose:
	  print("Loading data")

	db = util.Database(data_paths_fname)
	db.load_people()
	db.load_db(['loinc','loinc_vals','cpt','icd9_proc','icd9','ndc'])

	stats = util.read_yaml(stats_list_fname)[stats_key]

	if verbose:
		print("Calculating patient stats")

	data = ps.patient_stats(db, stats, stats_key, out_dir, stat_indices=None, verbose=verbose, check_if_file_exists=check_if_file_exists, save_files=True)

	if verbose:
		print("Building training data")

	outcome_data = btd.build_outcome_data(out_dir, outcome_fname)
	cohort_data = btd.setup(data_paths['demographics_fname'], outcome_fname, cohort_fname)
	# calc_gfr = True here because it's required to define the condition
	training_data = btd.build_training_data(db, cohort_data, gfr_loincs, lab_lower_bound, lab_upper_bound, \
		training_window_days, buffer_window_days, outcome_window_days, time_period_days, time_scale_days, gap_days, calc_gfr=True, verbose=verbose, \
		progression=progression, progression_lab_lower_bound=progression_lab_lower_bound, progression_lab_upper_bound=progression_lab_upper_bound, \
		progression_gap_days=progression_gap_days, progression_init_stages=progression_init_stages, progression_stages=progression_stages)
	training_data.to_csv(training_data_fname, index=False, sep='\t')


	features.features(db, training_data, feature_loincs, feature_diseases, feature_drugs, time_scale_days, features_fname, calc_gfr, verbose, add_age_sex)

	if split_fname is None:
		split_fname = out_dir + stats_key + '_split.txt'
		features.train_validation_test_split(training_data['person'].unique(), split_fname, verbose=verbose)

	features.split(features_fname, features_split_fname, split_fname, verbose)
	

	predict.predict(features_split_fname, lin_n_cv_iters, n_cv_iters, regularizations, n_labs, age_index, gender_index, predict_fname, nn_predict_fname)



model = load_model('ckd.model')
specific_gravity = input("Enter specific gravity: ") 
albumin = input("Enter albumin:")
serum_creatinine = input("Enter serum creatinine:")
hemoglobin = input("Enter hemoglobin:")
packed_cell_volume = input("Enter packed cell volume:")
hypertension = input("Enter hypertension:")
data = pd.DataFrame([{'sg': specific_gravity,'al': albumin,'sc': serum_creatinine,'hemo': hemoglobin,'pvc': packed_cell_volume,'htn': hypertension}])
print(data)
pred = model.predict(data)
pred = ["yes you have diagnosed with a chronic kidney disease" if y>=0.5 else "you are free from chronic kidney disease" for y in pred]
print(pred)
