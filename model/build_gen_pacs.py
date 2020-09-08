import pacs_architecture as pacs_architecture

def Generator():
    return pacs_architecture.Feature()


def Classifier(num_classes):
    return pacs_architecture.Predictor(num_classes)


def DomainPredictor(num_domains):
    return pacs_architecture.DomainPredictor(num_domains)
