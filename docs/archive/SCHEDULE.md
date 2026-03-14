# Project Schedule: Vision ML System

**⚠️ DEPRECATED**: This document has been superseded by [docs/ROADMAP.md](docs/ROADMAP.md)

Please refer to the roadmap for the current phase-based development plan.

---

## Original Schedule (Reference Only)

This document outlines a 12-week schedule for completing the Vision ML System project. This timeline is a suggestion and can be adapted as needed.

---

## Phase 1: Foundations (Weeks 1-2)

This phase is about setting up the basic infrastructure of our project.

### **Week 1: Project Setup, Git, Docker & Basic API**
*   **Goal:** Initialize the project, create a containerized "hello world" model API.
*   **Tasks:**
    *   Set up the project structure (`src`, `tests`, etc.).
    *   Initialize Git and push to GitHub/GitLab.
    *   Build a simple FastAPI or Flask app that returns a dummy prediction.
    *   Write a `Dockerfile` for the API.
    *   Use Docker Compose to run the API locally.
    *   Write a simple test for the API endpoint.

### **Week 2: Experiment Tracking & Data Versioning**
*   **Goal:** Set up experiment tracking and data versioning.
*   **Tasks:**
    *   Write a basic PyTorch training script.
    *   Integrate **MLflow** and **Comet-ML/TensorBoard** to log parameters and metrics.
    *   Set up **DVC** and connect it to a remote storage (e.g., a local directory or S3 bucket).
    *   Use DVC to track a dummy dataset and the trained model.

---

## Phase 2: Automation & CI/CD (Weeks 3-4)

This phase focuses on automating our workflows.

### **Week 3: CI/CD Pipelines**
*   **Goal:** Automate testing and Docker image builds.
*   **Tasks:**
    *   Set up a **GitHub Actions** workflow to run `pytest` on every push.
    *   Extend the workflow to build and push the Docker image to Docker Hub or another registry.
    *   (Optional) Replicate this setup using **Jenkins**, **GitLab CI/CD**, or **CircleCI**.

### **Week 4: Workflow Orchestration with Airflow**
*   **Goal:** Create an automated training pipeline.
*   **Tasks:**
    *   Set up a local Airflow environment.
    *   Create an Airflow DAG that orchestrates the training pipeline (e.g., data download, preprocessing, training, evaluation).
    *   (Optional) Explore using **Astro** for a managed Airflow experience.

---

## Phase 3: Deployment & Infrastructure (Weeks 5-7)

This phase is all about getting our model into a production-like environment.

### **Week 5: Kubernetes & Local Deployment**
*   **Goal:** Deploy the application to a local Kubernetes cluster.
*   **Tasks:**
    *   Install **Minikube** to run Kubernetes locally.
    *   Write Kubernetes manifest files (`deployment.yaml`, `service.yaml`).
    *   Deploy the containerized API to Minikube.
    *   Learn to use `kubectl` to manage the deployment.

### **Week 6: Cloud Deployment**
*   **Goal:** Deploy the model to a cloud platform.
*   **Tasks:**
    *   **GCP:** Deploy the application to **Google Cloud Run**.
    *   **AWS:** Deploy the model to **AWS SageMaker**.
    *   Compare the two experiences.

### **Week 7: GitOps with ArgoCD**
*   **Goal:** Implement a GitOps workflow for continuous deployment.
*   **Tasks:**
    *   Install **ArgoCD** on your Minikube cluster.
    *   Configure ArgoCD to monitor your GitHub repository for changes to the Kubernetes manifests.
    *   See how changes to the manifests in Git are automatically reflected in the cluster.

---

## Phase 4: Monitoring & Advanced Topics (Weeks 8-10)

This phase is about ensuring our system is reliable and exploring senior-level topics.

### **Week 8: System & Model Monitoring**
*   **Goal:** Add monitoring and observability to the system.
*   **Tasks:**
    *   Instrument the API with **Prometheus**.
    *   Set up **Grafana** and create a dashboard to visualize metrics.
    *   Use **Alibi-Detect** or **Evidently** to create a batch job that checks for data drift.

### **Week 9: Senior Topics I**
*   **Goal:** Explore feature stores and A/B testing.
*   **Tasks:**
    *   Set up a simple **Feast** feature store.
    *   Implement a simple A/B testing setup (e.g., using a reverse proxy to route traffic between two different model versions).

### **Week 10: Senior Topics II**
*   **Goal:** Explore streaming, security, and edge deployments.
*   **Tasks:**
    *   (Conceptual) Design a real-time inference system using **Kafka**.
    *   (Conceptual) Research and document potential security vulnerabilities in an ML system.
    *   (Optional) If you have the hardware, try deploying a model to a Raspberry Pi or NVIDIA Jetson.

---

## Phase 5: Buffer & Finalization (Weeks 11-12)

This phase is for catching up, documentation, and final presentation.

*   **Week 11:** Buffer week to catch up on any unfinished tasks or to dive deeper into a topic of interest.
*   **Week 12:** Finalize the project, clean up the code, write a comprehensive README, and prepare a presentation or blog post about your journey.
