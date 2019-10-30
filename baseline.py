# Usual imports
import numpy as np
#sns.set(rc={'figure.figsize': (9,6)})
#plt.rcParams['figure.figsize'] = (9,6)
from sklearn.naive_bayes import GaussianNB

from utils import *

# random seed for reproducibility
np.random.seed(42)

from sklearn.impute import SimpleImputer

df_train, df_lead, df_test, _, = get_cern_datasets()
imputer = SimpleImputer(strategy='median')
# prepare training data
train_data, train_target, train_weights = prepare_cern_dataset(df_train, True, imputer)
print("Len train data:", len(train_data))

# prepare testing data (private leaderboard dataset)
test_data, test_target, test_weights = prepare_cern_dataset(df_test, False, imputer)
print("Len private leaderboard (test) data:", len(test_data))

# prepare public leaderboard data
lead_data, lead_target, lead_weights = prepare_cern_dataset(df_lead, False, imputer)
print("Len public lead data:", len(lead_data))


# Baseline: Naive Bayes classifier with Gaussian
print("Gaussian naive Bayes classifier:\n")
model = GaussianNB()
model.fit(train_data, train_target)
# AMS scores on train, test set and public leaderboard
train_predicted = model.predict(train_data)
print("AMS Score on train:", ams_score(train_target, train_predicted,
                                       train_weights))

test_predicted = model.predict(test_data)
print("AMS Score on private leaderboard (test):", ams_score(test_target,
                                                            test_predicted,
                                                            test_weights))
lead_predicted = model.predict(lead_data)
print("AMS Score on public leaderboard:", ams_score(lead_target, lead_predicted,
                                                    lead_weights))

# AMS Score on train: 0.541116035855505
# AMS Score on private leaderboard (test): 0.7337585748518631
# AMS Score on public leaderboard: 0.3536717592993031


# AMS Score on train: 0.9739779997469158
# AMS Score on private leaderboard (test): 0.9909174142599658
# AMS Score on public leaderboard: 1.019830900722951
