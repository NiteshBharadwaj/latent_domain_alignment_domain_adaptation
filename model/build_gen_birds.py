import birds_architecture as birds_architecture

def Generator():
    return birds_architecture.Feature()


def Classifier(num_classes, inp_channels=256):
    return birds_architecture.Predictor(num_classes, inp_channels)


def DomainPredictor(num_domains, classwise=False, num_classes=None, classaware_dp=False):
    if classwise:
        return birds_architecture.DomainPredictor(num_domains * num_classes, classaware_dp=classaware_dp)
    else:
        return birds_architecture.DomainPredictor(num_domains, classaware_dp=classaware_dp)

