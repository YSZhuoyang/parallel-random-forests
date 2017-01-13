
import datetime
import arff, numpy as np
from sklearn.ensemble import RandomForestClassifier

START_IMPORT = datetime.datetime.now().replace(microsecond=0)

TRAIN_ARFF = arff.load(open('../Dataset/train/train-first200.arff', 'r'))
TRAIN_DATA = np.array(TRAIN_ARFF['data'])

TEST_ARFF = arff.load(open('../Dataset/test/dev-first200.arff', 'r'))
TEST_DATA = np.array(TEST_ARFF['data'])

START_TRAIN = datetime.datetime.now().replace(microsecond=0)

print("Time taken by loading data: ")
print(START_TRAIN - START_IMPORT)

FOREST = RandomForestClassifier(n_estimators=100, max_features=8, bootstrap=False, n_jobs=-1)
FOREST = FOREST.fit(TRAIN_DATA[0::, :-1], TRAIN_DATA[0::, -1])

END_TRAIN = datetime.datetime.now().replace(microsecond=0)

ACCURACY = FOREST.score(TEST_DATA[0::, :-1], TEST_DATA[0::, -1])

print("Accuracy: ")
print(ACCURACY)
print("Time taken by training: ")
print(END_TRAIN - START_TRAIN)
