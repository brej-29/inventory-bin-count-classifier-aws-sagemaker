<div align="center">
  <h1>📦 Inventory Monitoring via Bin Item Count Classification (Amazon Bin Images)</h1>
  <p><i>End-to-end Computer Vision capstone on AWS SageMaker — Baseline Training → HPO → Distributed Training (DDP) → Real-Time Endpoint → Lambda Integration → Defensive Cleanup</i></p>
</div>

<br>

<div align="center">
  <img alt="Language" src="https://img.shields.io/badge/Language-Python-blue">
  <img alt="Platform" src="https://img.shields.io/badge/Platform-Amazon%20SageMaker-232F3E?logo=amazonaws&logoColor=white">
  <img alt="Framework" src="https://img.shields.io/badge/Framework-PyTorch-ee4c2c?logo=pytorch&logoColor=white">
  <img alt="CV" src="https://img.shields.io/badge/Domain-Computer%20Vision-brightgreen">
  <img alt="Deployment" src="https://img.shields.io/badge/Inference-Real--Time%20Endpoint%20%2B%20Lambda-purple">
</div>

<div align="center">
  <br>
  <b>Built with:</b>
  <br><br>
  <code>Python</code> | <code>PyTorch</code> | <code>Torchvision</code> | <code>Amazon SageMaker</code> | <code>boto3</code> | <code>CloudWatch</code> | <code>AWS Lambda</code>
</div>

---

## **Overview**

In distribution centers, bins must contain an expected number of items for reliable fulfillment. Manual bin counting is slow, costly, and does not scale.  
This project builds an **image classification** model that predicts a **bin item count class (1–5)** from an input image, then deploys it as a **real-time SageMaker endpoint** and exposes inference via **AWS Lambda**.

This repository demonstrates production-style ML engineering on AWS:

- Data preparation and deterministic splits
- Baseline training (single instance)
- Hyperparameter tuning (HPO)
- Multi-instance distributed training (DDP)
- Real-time deployment and smoke testing
- Lambda-triggered inference (HTTP URL / S3 URI / API Gateway payloads)
- Defensive cleanup to avoid ongoing charges

---

## **Project Structure**

```
Inventory Management/
├── Documentation/
│   ├── Capstone Project Report.pdf
│   ├── Lambda Test Events Response.txt
│   └── proposal.pdf
├── Images/
│   ├── All_Training_Jobs.png
│   ├── Best_Model_Metrics.png
│   ├── CloudWatch_SS1.png
│   ├── CloudWatch_SS2.png
│   ├── CostExplorer_SS1.png
│   ├── CostExplorer_SS2.png
│   ├── Final_model.png
│   ├── Final_S3.png
│   ├── Hyperparameter_jobs.png
│   ├── Hyperparameter_training_jobs.png
│   ├── Lambda_API_Gateway_Test.png
│   ├── Lambda_Architecture.png
│   ├── Lambda_code.png
│   ├── Lambda_Function.png
│   ├── Lambda_HTTP_URL_Test.png
│   ├── Lambda_S3_URI_Test.png
│   ├── Project_Architecture.png
│   └── Sagemaker_models.png
├── local_eval/
│   ├── model/
│   │   ├── metrics.json
│   │   ├── model.pth
│   │   └── model.tar.gz
│   ├── confusion_matrix.png
│   └── metrics.json
├── file_list.json
├── final_inference.py
├── inference.py
├── Lambda.py
├── README.md
├── sagemaker.ipynb
├── train.py
└── train1.py
```
---

## **Key Files and What They Do**

### Notebook
- sagemaker.ipynb  
  The end-to-end orchestration notebook for: splitting → training → HPO → DDP → endpoint deploy → Lambda integration → cleanup.

### Training Entry Points
- train.py  
  Entry point for **single-instance training** (baseline + non-distributed workflow).
- train1.py  
  Entry point for **multi-instance distributed training (DDP)** (2 instances). Includes distributed-safe logging + checkpointing patterns.

### Inference Handlers
- inference.py  
  Inference handler used in the **initial deployment tests**.
- final_inference.py  
  Final inference handler used for the **final endpoint** (and Lambda integration). Returns a consistent JSON schema:
  predicted_label, predicted_index, confidence, probabilities, class_labels

### Lambda
- Lambda.py  
  AWS Lambda handler that:
  - Accepts request as HTTP image URL or S3 URI or API Gateway-style payload
  - Fetches image bytes
  - Invokes SageMaker endpoint (invoke_endpoint)
  - Returns prediction JSON response

---

## **Architecture**

### High-Level Pipeline (Training → Deployment → Lambda)

<img alt="Architecture Image" src="./Images/Project_Architecture.png" />

### Lambda Integration (How inference is triggered)

**Lambda Achitecture** 
<img alt="Architecture Image" src="./Images/Lambda_Architecture.png" />

**Lambda Function**   
<img alt="Architecture Image" src="./Images/Lambda_Function.png" />

**Lambda Function Code** 
<img alt="Architecture Image" src="./Images/Lambda_code.png" />

**Lambda HTTP URL Test**  
<img alt="Architecture Image" src="./Images/Lambda_HTTP_URL_Test.png" />

**Lambda S3 URI Test**  
<img alt="Architecture Image" src="./Images/Lambda_S3_URI_Test.png" />

**Lambda API Gateway Test** 
<img alt="Architecture Image" src="./Images/Lambda_API_Gateway_Test.png" />


### Monitoring and Cost Evidence

**CloudWatch Monitoring** 
<img alt="Architecture Image" src="./Images/CloudWatch_SS1.png" />

<img alt="Architecture Image" src="./Images/CloudWatch_SS2.png" />


**Cost Explorer** 
<img alt="Architecture Image" src="./Images/CostExplorer_SS1.png" />

<img alt="Architecture Image" src="./Images/CostExplorer_SS2.png" />


---

## **Problem Statement**

Given a bin image x, predict the item-count label y ∈ {1,2,3,4,5}, where y represents EXPECTED_QUANTITY.

This is a supervised **multi-class classification** problem:
- Input: JPG image
- Output: one of five count classes (1–5)

---

## **Evaluation Metrics**

This project tracks standard multi-class classification metrics:
- Accuracy
- Macro F1-score (primary)
- Macro Precision / Macro Recall
- Confusion Matrix

Macro metrics are emphasized because they treat each class equally and help reveal performance issues under class imbalance.

---

## **Results Summary (Quick View)**


| Stage | What it proves | Key Evidence |
|------|-----------------|--------------|
| Baseline (Single Instance) | End-to-end training + test evaluation works | Test Acc: 0.3203, Test Macro-F1: 0.3168 |
| HPO (Hyperparameter Tuning) | Objective improved via systematic search | Best objective (val_macro_f1): 0.387539 |
| DDP (2 instances) | Multi-instance training produces deployable artifact | Status: Completed, InstanceCount: 2, Artifact: s3://.../cbc-ddp-251225173021/output/model.tar.gz |
| Real-time Deploy | Endpoint is healthy and returns structured JSON | Endpoint InService + successful smoke test response |
| Lambda Integration | Serverless invocation works for multiple input types | HTTP URL / S3 URI / API Gateway payloads validated |
| Cleanup | No lingering inference resources | Endpoint + EndpointConfig deleted (defensive cleanup) |

---

## **Major Milestones Implemented**

1) Baseline training (train.py)  
2) HPO tuning job (objective metric: val_macro_f1)  
3) Distributed training on 2 GPU instances (train1.py)  
4) Real-time endpoint deployment + smoke test (final_inference.py)  
5) Lambda → SageMaker endpoint integration (Lambda.py)  
6) Defensive cleanup (endpoint, endpoint config, model objects)

---

## **Spot Training Attempt (Cost Optimization Note)**

Managed Spot Training was attempted to reduce cost, but the experiment was not a like-for-like comparison because:
- The baseline used a GPU instance (ml.g4dn.xlarge)
- The spot run used a CPU instance (ml.m5.2xlarge), which is dramatically slower for CNN training (ResNet50)
- The job hit MaxRuntime and was stopped (MaxRuntimeExceeded)

How to implement Spot Training properly:
- Use Spot with the SAME FAMILY of GPU instances as baseline (e.g., g4dn.xlarge) so runtime comparisons are meaningful
- Enable checkpointing (already done in this project) so interrupted spot runs can resume
- Increase MaxWaitTimeInSeconds and MaxRuntimeInSeconds appropriately for spot variability
- Keep training code resilient to restarts (idempotent data loading, safe checkpoint writes)

---

## **Getting Started (Local + SageMaker)**

### Prerequisites
- Python 3.10+
- AWS account with permissions for:
  - SageMaker (training, HPO, deployment)
  - S3 (read/write artifacts)
  - CloudWatch Logs
  - Lambda (create + invoke)
  - IAM roles (execution role for SageMaker and Lambda)
- Recommended environment: SageMaker Studio / Udacity workspace configured for SageMaker

### Install Dependencies
If you run locally:
- pip install sagemaker boto3 torch torchvision Pillow numpy pandas scikit-learn matplotlib

(If you run inside SageMaker-managed containers, many dependencies are already present.)

---

## **How to Run (Recommended Flow)**

### 1) Run the notebook end-to-end
Open:
- sagemaker.ipynb

Follow the cells in order:
- Data split + upload to S3
- Baseline training job (train.py)
- HPO job (train.py)
- DDP training job (train1.py)
- Deploy endpoint (final_inference.py)
- Smoke test invocation
- Create Lambda integration
- Cleanup (endpoint + endpoint config + model objects)

### 2) Confirm artifacts
Artifacts are stored in S3 and referenced in the notebook output logs.

---

## **Inference Payload Formats**

### Real-time endpoint (direct)
This project uses raw image bytes:
- Content-Type: application/x-image
- Body: image bytes

### Lambda invocation
Lambda supports multiple formats:
1) HTTP URL input:
   - { "image_url": "https://..." }

2) S3 URI input:
   - { "s3_uri": "s3://bucket/key.jpg" }

3) API Gateway style payload:
   - Standard event wrapper that contains a JSON body with one of the above.

Lambda fetches bytes, calls invoke_endpoint, and returns the model response JSON.

---

## **Cleanup (VERY IMPORTANT — Avoid Charges)**

After testing inference, delete resources:
- SageMaker Endpoint
- SageMaker EndpointConfig
- SageMaker Model objects (if created during repack/deploy)

This repository includes defensive cleanup logic in the notebook, and the project logs confirm endpoint and config deletion after validation.

---

## **Screenshots to Include (Suggested in README / Report)**

Minimum recommended:
- Images/All_Training_Jobs.png (proof of completed jobs)
- Images/Hyperparameter_training_jobs.png (HPO evidence)
- Images/Best_Model_Metrics.png or Images/Final_model.png (final model artifact / metrics)
- Images/CloudWatch_SS1.png + Images/CloudWatch_SS2.png (endpoint monitoring)
- Images/Lambda_Architecture.png + Images/Lambda_HTTP_URL_Test.png + Images/Lambda_S3_URI_Test.png (Lambda validation)
- Images/CostExplorer_SS1.png + Images/CostExplorer_SS2.png (cost evidence + cleanup discipline)

---

## **References (Copy-Paste List)**

Amazon SageMaker Distributed Training (Developer Guide)
https://docs.aws.amazon.com/sagemaker/latest/dg/distributed-training.html

Amazon SageMaker Training Toolkit (How SageMaker runs user scripts, env vars, MPI launch)
https://github.com/aws/sagemaker-training-toolkit

Amazon SageMaker Hyperparameter Tuning (Developer Guide)
https://docs.aws.amazon.com/sagemaker/latest/dg/automatic-model-tuning.html

SageMaker API: CreateHyperParameterTuningJob
https://docs.aws.amazon.com/sagemaker/latest/APIReference/API_CreateHyperParameterTuningJob.html

SageMaker API: CreateTrainingJob (EnableManagedSpotTraining / MaxRuntimeInSeconds / MaxWaitTimeInSeconds)
https://docs.aws.amazon.com/sagemaker/latest/APIReference/API_CreateTrainingJob.html

Troubleshoot Real-Time Inference / Deployment (Health checks, common failures)
https://docs.aws.amazon.com/sagemaker/latest/dg/deploy-model-troubleshoot.html

SageMaker Hosting: Endpoints (Concepts)
https://docs.aws.amazon.com/sagemaker/latest/dg/how-it-works-hosting.html

boto3 SageMaker Runtime: invoke_endpoint
https://boto3.amazonaws.com/v1/documentation/api/latest/reference/services/sagemaker-runtime/client/invoke_endpoint.html

AWS Lambda Python handler basics
https://docs.aws.amazon.com/lambda/latest/dg/python-handler.html

Torchvision ResNet model documentation
https://pytorch.org/vision/stable/models/resnet.html

Torchvision model weights enums (pretrained → weights migration)
https://pytorch.org/vision/stable/models.html

Torchvision ImageFolder dataset
https://pytorch.org/vision/stable/generated/torchvision.datasets.ImageFolder.html

PyTorch tutorial: Saving and loading models (state_dict, checkpoints)
https://pytorch.org/tutorials/beginner/saving_loading_models.html

Amazon CloudWatch Logs (concepts)
https://docs.aws.amazon.com/AmazonCloudWatch/latest/logs/WhatIsCloudWatchLogs.html

---

## **Author / Contact**

Brejesh Balakrishnan  
LinkedIn: https://www.linkedin.com/in/brejesh-balakrishnan-7855051b9/

If you have questions, suggestions, or want to discuss improvements (better backbones, augmentation, focal loss, class weighting, and stronger evaluation), feel free to connect on LinkedIn.
