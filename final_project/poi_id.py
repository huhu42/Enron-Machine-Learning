#!/usr/bin/python

import pickle
import sys

sys.path.append("../tools/")

from feature_format import featureFormat, targetFeatureSplit
from tester import dump_classifier_and_data
import pprint


### Task 1: Select what features you'll use.
### features_list is a list of strings, each of which is a feature name.
### The first feature must be "poi".

### Load the dictionary containing the dataset
with open("final_project_dataset.pkl", "r") as data_file:
    data_dict = pickle.load(data_file)


### Task 2: Remove outliers - remove outlier given by enron_outliers.py file
print len(data_dict)
data_dict.pop("TOTAL", 0)
### also another one by looking through the PDF
data_dict.pop("THE TRAVEL AGENCY IN THE PARK", 0)
### Checking to make sure exactly 2 entries are removed
print len(data_dict)

### Task 3: Create new feature(s)

def dict_to_list(key,normalizer):
    new_list=[]

    for i in data_dict:
        if data_dict[i][key]=="NaN" or data_dict[i][normalizer]=="NaN":
            new_list.append(0.)
        elif data_dict[i][key]>=0:
            new_list.append(float(data_dict[i][key])/float(data_dict[i][normalizer]))
    return new_list

### create two lists of new features
from_poi_email_fraction=dict_to_list("from_poi_to_this_person","to_messages")
to_poi_email_fraction=dict_to_list("from_this_person_to_poi","from_messages")

### insert new features into data_dict
count=0
for i in data_dict:
    data_dict[i]["from_poi_email_fraction"]=from_poi_email_fraction[count]
    data_dict[i]["to_poi_email_fraction"]=to_poi_email_fraction[count]
    count +=1

#picked the top 10 features with SelectKbest, then tuned the features by taking away 1 feature at a time, till I get the highest accuracy
features_list = ['poi','exercised_stock_options', 'total_stock_value', 'bonus'\
                 #,'salary'
                #'to_poi_email_fraction'\
                #'deferred_income'\
                 #'long_term_incentive'\
                 #'restricted_stock'\
                 #'total_payments'\
                 #'shared_receipt_with_poi'\
                     ]

# picking the top 10 features
#pprint.pprint (Select_K_Best(data_dict, features_list, 10))

### Store to my_dataset for easy export below.
my_dataset = data_dict

data = featureFormat(data_dict, features_list)

### Extract features and labels from dataset for local testing
data = featureFormat(my_dataset, features_list, sort_keys = True)
labels, features = targetFeatureSplit(data)

### Scale features
from sklearn.preprocessing import MinMaxScaler

scaler = MinMaxScaler()
features = scaler.fit_transform(features)

### Task 4: Try a varity of classifiers
### Please name your classifier clf for easy export below.
### Note that if you want to do PCA or other multi-stage operations,
### you'll need to use Pipelines. For more info:
### http://scikit-learn.org/stable/modules/pipeline.html

# Provided to give you a starting point. Try a variety of classifiers.
from sklearn.cross_validation import train_test_split
features_train, features_test, labels_train, labels_test = \
    train_test_split(features, labels, test_size=0.3, random_state=42)
#Testing out Naive Bayes but decided not to use because Decision Tree offered better recall
from sklearn.naive_bayes import GaussianNB
clf1 = GaussianNB()
clf1 = clf1.fit(features_train, labels_train)
pred1 = clf1.predict(features_test)

from tester import test_classifier
print test_classifier(clf1, my_dataset, features_list, folds = 1000)

### Task 6: Dump your classifier, dataset, and features_list so anyone can
### check your results. You do not need to change anything below, but make sure
### that the version of poi_id.py that you submit can be run on its own and
### generates the necessary .pkl files for validating your results.


### Task 5: Tune your classifier to achieve better than .3 precision and recall 
### using our testing script. Check the tester.py script in the final project
### folder for details on the evaluation method, especially the test_classifier
### function. Because of the small size of the dataset, the script uses
### stratified shuffle split cross validation. For more info: 
### http://scikit-learn.org/stable/modules/generated/sklearn.cross_validation.StratifiedShuffleSplit.html

# Example starting point. Try investigating other evaluation techniques!

from sklearn.tree import DecisionTreeClassifier
from sklearn.grid_search import GridSearchCV
from sklearn.grid_search import RandomizedSearchCV
from sklearn.cross_validation import cross_val_score

dct = DecisionTreeClassifier(criterion = 'gini', min_samples_split = 2, max_depth = None, min_samples_leaf = 1)

param_grid = {#"criterion": ["gini", "entropy"],
              #"min_samples_split": [2, 10, 20],
              #"max_depth": [None, 2, 5, 10],
              #"min_samples_leaf": [1, 5, 10],
              "max_leaf_nodes": [None, 5, 10, 20],
              }
clf = GridSearchCV(dct, param_grid, cv = 5, scoring = "recall")
clf.fit(features_train, labels_train)


print "Best Params: "
pprint.pprint (clf.best_params_)

print test_classifier(clf, my_dataset, features_list, folds = 1000)

dump_classifier_and_data(clf, my_dataset, features_list)
