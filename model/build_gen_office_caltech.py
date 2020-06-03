import office_caltech_architecture as office_caltech_architecture
import office_caltech_architecture101 as office_caltech_architecture101

def Generator():
    #return office_caltech_architecture.Feature()
    return office_caltech_architecture101.Feature()


def Classifier(num_classes):
    #return office_caltech_architecture.Predictor(num_classes)
    return office_caltech_architecture101.Predictor(num_classes)

def DomainPredictor(num_domains):
    #return office_caltech_architecture.DomainPredictor(num_domains)
    return office_caltech_architecture101.DomainPredictor(num_domains)