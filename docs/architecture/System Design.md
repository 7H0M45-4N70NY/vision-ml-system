# Vision ML System - High-Level System Design

## 1. Executive Summary

This document outlines the architectural transformation of the Vision ML System from a monolithic application to a decoupled, scalable, event-driven architecture. Following FAANG-level engineering principles, the system is designed to handle real-time computer vision inference with high throughput, while asynchronously managing model drift detection, deep auditing (dual detectors), and continuous learning loops.

## 2. Architectural Principles

We adopt a **Microservices-based, Event-Driven Architecture** to ensure:

*   **Decoupling:** Separation of ingestion, inference, analytics, and storage allows independent scaling and maintenance.
*   **Resilience:** Failures in the heavy "overhead" audit systems do not block the real-time primary inference.
*   **Scalability:** The primary inference engine can scale horizontally based on stream load, distinct from the resource-intensive secondary detectors.
*   **Observability:** Real-time drift detection and system health monitoring are built-in first-class citizens.

## 3. High-Level Architecture

```mermaid
graph TD
    subgraph "Edge / Ingestion Layer"
        Cam[Camera Feeds / RTSP] --> Ingest[Ingestion Service]
        Ingest -->|Raw Frames| HotStream[(Hot Stream / Redis / NATS)]
    end

    subgraph "Real-Time Inference Layer (Primary)"
        HotStream --> PrimaryDet[Primary Detector Service (YOLO)]
        PrimaryDet -->|Detections & Confidence| ResultStream[(Result Stream / Kafka)]
    end

    subgraph "Overhead / Audit Layer (Secondary)"
        ResultStream --> DriftService[Drift Detection Service]
        DriftService -->|Low Confidence / Trigger| SecondaryDet[Secondary Detector Service (RF-DETR)]
        SecondaryDet -->|High Quality Labels| AutoLabelDB[(Auto-Label Storage)]
        
        Note1[Decoupled from Critical Path] --- DriftService
    end

    subgraph "Data & Analytics Layer"
        ResultStream --> Analytics[Analytics Aggregator]
        Analytics --> TimeSeriesDB[(TimescaleDB / Influx)]
        Analytics --> Dashboard[Streamlit / Grafana]
    end

    subgraph "MLOps & Continuous Learning"
        AutoLabelDB --> Training[Training Pipeline]
        Training --> ModelRegistry[Model Registry]
        ModelRegistry -->|Deploy| PrimaryDet
    end
```

## 4. Detailed Component Design

### 4.1. Ingestion Service (The Gateway)
*   **Responsibility:** Connects to RTSP streams, webcams, or video files.
*   **Mechanism:** Decodes video frames and pushes them to a low-latency shared memory buffer or message bus (e.g., Redis Streams or NATS JetStream).
*   **Optimization:** Performs resizing/normalization at the edge to reduce bandwidth if necessary.

### 4.2. Primary Inference Service (Fast Path)
*   **Responsibility:** Real-time object detection and tracking.
*   **Engine:** Lightweight, high-throughput models (e.g., YOLOv11n).
*   **Design:** Stateless workers consuming frames from the `HotStream`.
*   **Output:** Publishes detection metadata (bounding boxes, class IDs, confidence scores) to the `ResultStream`. It does *not* wait for drift checks.

### 4.3. Drift Detection & Overhead Audit Service (The "Dual Detector")
*   **Concept:** Instead of a blocking dual-detector inside the main loop, this is an asynchronous "Overhead" service.
*   **Drift Detection:** Consumes the `ResultStream`. Monitors distribution of confidence scores and class frequencies (KL Divergence, Rolling Mean).
*   **Trigger Mechanism:**
    1.  **Statistical Drift:** If average confidence drops below threshold $T$ over window $W$.
    2.  **Point Anomalies:** Individual low-confidence detections.
    3.  **Random Sampling:** Randomly samples 1% of frames for audit.
*   **Secondary Detector (Oracle):** When triggered, pulls the corresponding raw frame and runs the heavy, accurate model (RF-DETR).
*   **Outcome:** If the Secondary Detector finds objects the Primary missed, the frame is tagged and stored for **Active Learning**.

### 4.4. Analytics & Storage
*   **Analytics Aggregator:** Consumes detection events to calculate business metrics (Visitor Count, Dwell Time).
*   **Hot Storage:** Redis for current state (last 1 minute of data).
*   **Cold Storage:** S3/MinIO for saved frames (drift events) and auto-labeled datasets.
*   **Database:** PostgreSQL/TimescaleDB for historical analytics and drift logs.

## 5. Implementation Strategy (Refactoring Plan)

### Phase 1: Decouple Interface from Logic (✅ Completed)
*   **Refactored**: `InferencePipeline` moved to `src/vision_ml/inference`.
*   **API Layer**: Created FastAPI service (`src/vision_ml/api`) to handle requests.
*   **Dockerization**: Fully containerized application with `docker-compose`.
*   **Testing**: Added CI/CD pipelines and unit tests for core components.

### Phase 2: Asynchronous Drift Detection (🚧 Next Up)
*   Extract `DriftDetector` and `DualDetector` logic from the critical `process_frame` loop.
*   Implement a background thread or separate process to handle "Audit" tasks.
*   The primary pipeline merely "fires and forgets" frames to the audit queue.

### Phase 3: Service Separation (Future)
*   Deploy `Primary Detector` and `Overhead Detector` as separate containers.
*   Implement standard protocols (gRPC or HTTP/2) for inter-service communication.
*   Add Prometheus/Grafana for real-time system monitoring.

## 6. Key Advantages of this Design

1.  **Zero Latency Impact:** The heavy RF-DETR model never blocks the live video feed. It runs at its own pace on a separate queue.
2.  **Resource Efficiency:** We don't need GPU resources for the secondary detector on every frame, only when drift is suspected or for sampling.
3.  **Robustness:** If the Audit service crashes, the camera feed and primary detection continue uninterrupted.
4.  **Data-Centric AI:** The system automatically curates the "hardest" examples (where Primary and Secondary disagree) for the next training cycle.
