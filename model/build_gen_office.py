import svhn2mnist
import usps
import syn2gtrsb
#import syndig2svhn
import office_architecture_resnet50 as office_architecture_new
import office_architecture_alex

def Generator():
    # #if source == 'usps' or target == 'usps':
    #     return usps.Feature()
    # elif source == 'svhn':
    return office_architecture_new.Feature()


def Classifier(num_classes):
    # if source == 'usps' or target == 'usps':
    #     return usps.Predictor()
    # if source == 'svhn':
    return office_architecture_new.Predictor(num_classes)

def DomainPredictor(num_domains):
    return office_architecture_new.DomainPredictor(num_domains)

def GeneratorAlex():
    # #if source == 'usps' or target == 'usps':
    #     return usps.Feature()
    # elif source == 'svhn':
    return office_architecture_alex.Feature()


def ClassifierAlex(num_classes):
    # if source == 'usps' or target == 'usps':
    #     return usps.Predictor()
    # if source == 'svhn':
    return office_architecture_alex.Predictor(num_classes)

def DomainPredictorAlex(num_domains):
    return office_architecture_alex.DomainPredictor(num_domains)
