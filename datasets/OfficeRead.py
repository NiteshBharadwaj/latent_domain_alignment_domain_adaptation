import csv
def read_office_domain(domain):
	trainFile = '/home/kujain/DomainAdaptation/code_MSDA_digit/data/office_sample/' + domain + '_data_train.csv'
	testFile = '/home/kujain/DomainAdaptation/code_MSDA_digit/data/office_sample/' + domain + '_data_test.csv'
	validFile = '/home/kujain/DomainAdaptation/code_MSDA_digit/data/office_sample/' + domain + '_data_valid.csv'

	trainReader = csv.reader(open(trainFile, 'r'))
	testReader = csv.reader(open(testFile, 'r'))
	validReader = csv.reader(open(validFile, 'r'))

	paths_train, labels_train, paths_test, labels_test, paths_valid, labels_valid = [], [], [], [], [], []

	for row in trainReader:
		paths_train.append('/home/kujain/DomainAdaptation/code_MSDA_digit'+row[2][1:])
		labels_train.append(int(row[1]))

	for row in testReader:
		paths_test.append('/home/kujain/DomainAdaptation/code_MSDA_digit'+row[2][1:])
		labels_test.append(int(row[1]))

	for row in validReader:
		paths_valid.append('/home/kujain/DomainAdaptation/code_MSDA_digit'+row[2][1:])
		labels_valid.append(int(row[1]))

	return paths_train, labels_train, paths_test, labels_test, paths_valid, labels_valid