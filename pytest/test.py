
import arff, numpy as np
from sklearn.ensemble import RandomForestClassifier 

train_arff = arff.load(open('../Dataset/train/train-first200.arff', 'r'))
train_data = np.array(train_arff['data'])

test_arff = arff.load(open('../Dataset/test/dev-first200.arff', 'r'))
test_data = np.array(test_arff['data'])


forest = RandomForestClassifier(n_estimators = 100, max_features = 8)
forest = forest.fit(train_data[0::, :-1],train_data[0::, -1])

accuracy = forest.score(test_data[0::, :-1],test_data[0::, -1])

print( accuracy )

#train_data = [[float(string) for string in inner] for inner in train_data[0::, 0:-1]]
#s = sum(map(sum, train_data))

#l = len( train_data[0] ) - 1

#print( s )
