1. Create a Pod with Queue Container with port 8161 and add Service and Expose 30010
```sh
kubectl get all
kubectl describe pod/pod-queue
kubectl port-forward pod/pod-queue 30010:8161

# username: admin
# password: admin 
```