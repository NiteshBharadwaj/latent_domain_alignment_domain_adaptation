import compcars_architecture
import compcars_architecture_new


def Classifier(num_classes):
    # return compcars_architecture.Predictor_ResNet18(num_classes)
    return compcars_architecture_new.Predictor(num_classes)


def Generator():
    # return compcars_architecture.Feature_ResNet18()
    return compcars_architecture_new.Feature()


def DomainPredictor(num_domains, aux_classes):
    # return compcars_architecture.DomainPredictor_ResNet18(num_domains)
    return compcars_architecture_new.DomainPredictor_ResNet18(num_domains, aux_classes)
