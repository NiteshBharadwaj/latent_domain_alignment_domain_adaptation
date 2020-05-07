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
  ./deploy --name temp-pod --ngpus 0 --type simple_pod
```

To check
```
 kubectl logs [pod-name]
```
Finally, important when you're done to
```
kubectl delete pod [pod name]
```


Whenever you want to run a job 
Warning: You should not create any infinite loop in a job (atleast by choice). Manjot and his team was banned for this.
```
  cd deployments
  ./deploy --name [job name] --ngpus 1 --type deeplearning_job --meml 16Gi --memr 16Gi --command "[CMD1;CMD2]"
```
Example:
```
  ./deploy --name latent-office-job --ngpus 1 --type deeplearning_job --meml 8Gi --memr 8Gi --command "bash ./experiment_do.sh da 1500 0 record_office/dslr_amazon_soft_cluster_8_1 soft_cluster office 8 0.02 1 /localdata/office/ no"
```

If you want to install new packages or unzip files to local before starting a job or pod, checkout deploy.py


GPU Monitoring:
https://grafana.nautilus.optiputer.net/d/dRG9q0Ymz/k8s-compute-resources-namespace-gpus?orgId=1&refresh=30s&var-namespace=mc-lab
https://grafana.nautilus.optiputer.net/d/85a562078cdf77779eaa1add43ccec1e/kubernetes-compute-resources-namespace-pods?orgId=1&refresh=10s&var-datasource=default&var-cluster=&var-namespace=mc-lab

Overall, the pods and jobs that we create should be working at >50% capacity. Otherwise we get warning and ban.

Other commands backup
```
    kubectl config set-context mc-lab --namespace=mc-lab --cluster=nautilus --user=[copy from other contexts]
    kubectl config use-context mc-lab
```

