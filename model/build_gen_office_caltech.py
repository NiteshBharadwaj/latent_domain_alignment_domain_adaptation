import office_caltech_alexnet_architecture as office_caltech_architecture
import office_caltech_architecture as office_caltech_architecture


def Generator():
    return office_caltech_architecture.Feature()


def Classifier(num_classes):
    return office_caltech_architecture.Predictor(num_classes)


def DomainPredictor(num_domains):
    return office_caltech_architecture.DomainPredictor(num_domains)
