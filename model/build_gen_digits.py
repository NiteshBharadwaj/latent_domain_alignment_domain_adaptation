import svhn2mnist
import usps
import syn2gtrsb
#import syndig2svhn


def Generator():
    return svhn2mnist.Feature()


#    return svhn2mnist.Feature_ResNet18()


def Classifier():
    return svhn2mnist.Predictor()


#    return svhn2mnist.Predictor_ResNet18()

def DomainPredictor(num_domains):
    return svhn2mnist.DomainPredictor(num_domains)
#    return svhn2mnist.DomainPredictor_ResNet18(num_domains)
