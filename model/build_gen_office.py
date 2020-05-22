import office_architecture_new


def Classifier(num_classes):
    return office_architecture_new.Predictor(num_classes)


def Generator():
    return office_architecture_new.Feature()


def DomainPredictor(num_domains):
    return office_architecture_new.DomainPredictor(num_domains)
