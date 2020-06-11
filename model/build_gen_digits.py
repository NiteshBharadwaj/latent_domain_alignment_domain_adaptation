import svhn2mnist
import usps
import svhn2mnist_cd


def Generator(cd=False, usps_only=False):
    if usps_only:
        # Since the dataset is small, we use a smaller network here
        return usps.Feature()
    if cd:
        return svhn2mnist_cd.Feature()
    else:
        return svhn2mnist.Feature()


def Classifier(cd=False, usps_only=False):
    if usps_only:
        return usps.Predictor()
    if cd:
        return svhn2mnist_cd.Predictor()
    else:
        return svhn2mnist.Predictor()


def DomainPredictor(num_domains, cd=False, usps_only=False):
    if usps_only:
        return usps.DomainPredictor(num_domains)
    if cd:
        return svhn2mnist_cd.DomainPredictor(num_domains)
    else:
        return svhn2mnist.DomainPredictor(num_domains)
