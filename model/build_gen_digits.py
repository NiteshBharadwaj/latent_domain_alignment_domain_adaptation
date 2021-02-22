import svhn2mnist
import usps
import syn2gtrsb
#import syndig2svhn
import svhn2mnist_cd
def Generator(cd=False, usps_only=False):
    if usps_only:
        #Since the dataset is small, we use a smaller network here
        return usps.Feature()
    if cd:
        return svhn2mnist_cd.Feature()
    else:
        return svhn2mnist.Feature()
#    return svhn2mnist.Feature_ResNet18()


def Classifier(cd=False, usps_only=False):
    if usps_only:
        return usps.Predictor()
    if cd:
        return svhn2mnist_cd.Predictor()
    else:
        return svhn2mnist.Predictor()
#    return svhn2mnist.Predictor_ResNet18()

def DomainPredictor(num_domains,cd=False, usps_only=False, classwise=False, num_classes=None):
    if usps_only:
        return usps.DomainPredictor(num_domains*num_classes)
    if cd:
        return svhn2mnist_cd.DomainPredictor(num_domains*num_classes)
    else:
        return svhn2mnist.DomainPredictor(num_domains*num_classes)
#    return svhn2mnist.DomainPredictor_ResNet18(num_domains)
