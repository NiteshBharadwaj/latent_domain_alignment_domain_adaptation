import domainnet_architecture as domainnet_architecture

def Generator():
    return domainnet_architecture.Feature()


def Classifier(num_classes):
    return domainnet_architecture.Predictor(num_classes)


def DomainPredictor(num_domains):
    return domainnet_architecture.DomainPredictor(num_domains)
