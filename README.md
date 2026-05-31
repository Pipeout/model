# Model Training Repository

This repository is exclusively responsible for **model training and experiment execution**. It contains the code required to train machine learning models, log experiments, and generate model artifacts.

The repository is maintained with **two distinct branches**, each targeting a different environment and deployment workflow.

---

# `main` Branch — Local Development

The `main` branch is intended for local development, experimentation, and debugging.

In this setup:

* Training jobs run locally using Docker
* Data is read from Amazon S3
* Model artifacts can be stored locally or uploaded to S3
* AWS credentials are configured through a local `.env` file

## AWS Credentials

To access datasets and artifact storage in S3, configure your AWS credentials in a `.env` file.

Example:

```env
AWS_ACCESS_KEY_ID=your_access_key
AWS_SECRET_ACCESS_KEY=your_secret_key
AWS_DEFAULT_REGION=us-east-1
```

## Security Recommendations

Sensitive files should never be committed to the repository.

Ensure the following files are included in both `.gitignore` and `.dockerignore`:

```gitignore
.env
```

---

# `dev-prod` Branch — Production Environment

The `dev-prod` branch contains the production-ready configuration used for training models in AWS.

This branch uses:

* Amazon ECS
* Amazon ECR
* GitHub Actions CI/CD
* AWS OIDC Authentication

Unlike the `main` branch, production deployments do not rely on static AWS credentials.

Training jobs are executed inside ECS tasks using container images stored in Amazon ECR.

---

# CI/CD Pipelines

## `main` Branch CI/CD

The CI/CD pipeline for `main`:

1. Builds the training image
2. Pushes the image to Docker Hub

### Required GitHub Secrets

```text
DOCKERHUB_USERNAME
DOCKERHUB_TOKEN
```

---

## `dev-prod` Branch CI/CD

The CI/CD pipeline for `dev-prod`:

1. Builds the training image
2. Pushes the image to Amazon ECR
3. Updates ECS task definitions
4. Deploys the latest training image

This pipeline authenticates with AWS using OIDC.

### Required GitHub Secrets / Variables

```text
AWS_REGION
AWS_ARN_ROLE
```

* `AWS_REGION`: AWS region where the infrastructure is deployed.
* `AWS_ARN_ROLE`: IAM role assumed by GitHub Actions through OIDC.

---

# Runtime Environment Variables

Training containers receive their runtime configuration through ECS task definitions.

Common variables include:

```text
S3_BUCKET_NAME
MLFLOW_TRACKING_URI
AWS_REGION
```

Additional variables may be injected depending on the training workflow and infrastructure configuration.

---

# Infrastructure as Code (IaC)

Infrastructure resources such as ECS clusters, ECR repositories, networking, IAM roles, and task definitions are managed separately through the Infrastructure as Code repository.

For infrastructure configuration details, refer to the corresponding [IaC repository](https://github.com/Pipeout/IaC/blob/main/ecs_task_definitions.tf).

---

# Notes

* The `main` branch is intended for local experimentation and model development.
* The `dev-prod` branch is intended for automated production training workloads in AWS.
* This repository does not contain deployment or inference services; its sole responsibility is training machine learning models.
* Environment-specific configuration should be provided through GitHub Actions secrets, ECS task definitions, or AWS services rather than hardcoded values.

