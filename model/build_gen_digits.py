import svhn2mnist
import usps
import syn2gtrsb
#import syndig2svhn

def Generator():
    # #if source == 'usps' or target == 'usps':
    #     return usps.Feature()
    # elif source == 'svhn':
    return svhn2mnist.Feature()
#    return svhn2mnist.Feature_ResNet18()


def Classifier():
    # if source == 'usps' or target == 'usps':
    #     return usps.Predictor()
    # if source == 'svhn':
    return svhn2mnist.Predictor()
#    return svhn2mnist.Predictor_ResNet18()

def DomainPredictor(num_domains):
    return svhn2mnist.DomainPredictor(num_domains)
#    return svhn2mnist.DomainPredictor_ResNet18(num_domains)
