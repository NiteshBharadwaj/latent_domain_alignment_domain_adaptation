One-time Settings:
Download Kubectl https://kubernetes.io/docs/tasks/tools/install-kubectl/
Login to https://nautilus.optiputer.net/ using SSO
Download config by clicking get config at the top. Place it in ~/.kube/ folder (On Ubuntu, otherwise in your corresponding Kube folder)
Check:
```
    kubectl get pods
```
This should run without error.


Whenever you want to run a pod (Contact nitesh if you are unsure of difference between pod and job):
```
  cd deployments
  ./deploy --name [pod-name] --ngpus 1 --type simple_pod
  kubectl get pods
  kubectl exec -it [pod-name] /bin/bash 
```
```
  ./deploy --name temp-pod --ngpus 1 --type simple_pod
```

To check
```
 kubectl logs [pod-name]
```


Whenever you want to run a job 
Warning: You should not create any infinite loop in a job (atleast by choice). Manjot and his team was banned for this.
```
  cd deployments
  ./deploy --name [job name] --ngpus 1 --type deeplearning_job --meml 16Gi --memr 16Gi --command "[COMMAND HERE]"
```
Example:
```
  ./deploy --name latent-digit-job --ngpus 1 --type deeplearning_job --meml 16Gi --memr 16Gi --command " bash experiment_do.sh mnistm 100 0 record/mnistm soft_cluster digits 4 yes"
```

If you want to install new packages before starting a job or pod, check the commands in deploy.py



Other commands backup
```
    kubectl config set-context mc-lab --namespace=mc-lab --cluster=nautilus --user=[copy from other contexts]
    kubectl config use-context mc-lab
```

