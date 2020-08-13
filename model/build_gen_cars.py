import compcars_architecture
import compcars_architecture_new


def Classifier(num_classes):
    # return compcars_architecture.Predictor_ResNet18(num_classes)
    return compcars_architecture_new.Predictor(num_classes)


def Generator(is_fine_tune=False, extra_fc=False):
    # return compcars_architecture.Feature_ResNet18()
    return compcars_architecture_new.Feature(is_fine_tune, extra_fc)


def DomainPredictor(num_domains, aux_classes, is_fine_tune=False, extra_fc=False, weight_reinit=False):
    # return compcars_architecture.DomainPredictor_ResNet18(num_domains)
    return compcars_architecture_new.DomainPredictor_ResNet18(num_domains, aux_classes, is_fine_tune, extra_fc, weight_reinit)
