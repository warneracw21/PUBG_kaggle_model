
PATH_TO_PROJECT = '/Users/andrewwarner/Developer/NDL/kaggle/PUBG/'

# data_provider.py
TRAIN_PATH = PATH_TO_PROJECT + 'data/train.csv'
EVAL_PATH = PATH_TO_PROJECT + 'data/eval.csv'

BATCH_SIZE = 50

COLUMN_NAMES = ['assists', 'boosts', 'damageDealt', 'DBNOs',
       			'headshotKills', 'heals', 'killPlace', 'killPoints', 'kills',
       			'killStreaks', 'longestKill', 'maxPlace', 'numGroups', 'revives',
       			'rideDistance', 'roadKills', 'swimDistance', 'teamKills',
       			'vehicleDestroys', 'walkDistance', 'weaponsAcquired', 'winPoints',
       			'winPlacePerc']

FIELD_DEFAULTS = [0, 0, 0, 0.0, 0, 0, 0, 0, 0, 
				  0, 0, 0.0, 0, 0, 0, 0.0, 
				  0, 0.0, 0, 0, 0.0, 0, 0, 0.0]
FIELD_DEFAULTS = [[k] for k in FIELD_DEFAULTS]



