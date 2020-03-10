import svhn2mnist
import usps
import syn2gtrsb
#import syndig2svhn
import compcars_architecture

def Generator():
    # #if source == 'usps' or target == 'usps':
    #     return usps.Feature()
    # elif source == 'svhn':
    return compcars_architecture.Feature_ResNet18()


def Classifier(num_classes):
    # if source == 'usps' or target == 'usps':
    #     return usps.Predictor()
    # if source == 'svhn':
    return compcars_architecture.Predictor_ResNet18(num_classes)

def DomainPredictor(num_domains):
    return compcars_architecture.DomainPredictor_ResNet18(num_domains)
