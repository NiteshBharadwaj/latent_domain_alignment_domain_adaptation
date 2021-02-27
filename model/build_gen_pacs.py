import pacs_architecture as pacs_architecture

def Generator():
    return pacs_architecture.Feature()


def Classifier(num_classes):
    return pacs_architecture.Predictor(num_classes)


def DomainPredictor(num_domains, classwise=False, num_classes=None):
    if classwise:
        return pacs_architecture.DomainPredictor(num_domains*num_classes)
    else:
        return pacs_architecture.DomainPredictor(num_domains)

