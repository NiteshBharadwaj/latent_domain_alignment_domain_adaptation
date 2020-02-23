import svhn2mnist
import usps
import syn2gtrsb
#import syndig2svhn

def Generator():
    # #if source == 'usps' or target == 'usps':
    #     return usps.Feature()
    # elif source == 'svhn':
    return svhn2mnist.Feature()


def Classifier():
    # if source == 'usps' or target == 'usps':
    #     return usps.Predictor()
    # if source == 'svhn':
    return svhn2mnist.Predictor()

def DomainPredictor(num_domains):
    return svhn2mnist.DomainPredictor(num_domains)
