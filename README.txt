# Prerequisites

- OSSCI access

# Deploy Poro2 as an inference service (port 8000)

1. Apply the Kubernetes manifest:

    ```bash
    kubectl apply -f deployment-vllm.yaml -n silo-gen-models
    ```

2. Obtain the pod name:

    ```bash
    kubectl get pods -n silo-gen-models -l app=poro2-vllm-api
    ```

3. Then, open port forwarding to access the service locally:

    ```bash
    kubectl port-forward -n silo-gen-models pod/<POD-NAME> 8000:8000
    ```

4. Once port-forwarded, the model can be tested with a simple curl command:

    ```bash
    # Health check
    curl.exe -X GET http://localhost:8000/health
    

# Chat request
    curl.exe -X POST http://localhost:8000/v1/chat/completions -H "Content-Type: application/json" -d '{"messages": [ {"role": "user", "content": "Kerro minulle Suomen historiasta."} ],  "max_new_tokens": 300, "temperature": 0.7 }'

