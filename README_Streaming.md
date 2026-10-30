# Real-Time Streaming & Audit Pipeline Extension

An event-driven streaming extension for the Churn Prediction Engine. This module handles real-time customer telemetry ingestion via Apache Kafka, processes events through a containerized consumer worker using XGBoost and k-Means, commits audit logs to a persistent SQLite database, and visualizes live metrics via an interactive Streamlit dashboard.

---

##System Architecture

[ stream_producer.py ] ---> ( Kafka Topic: customer-events )
│
▼
[ Dockerized Consumer Worker ]
├── XGBoost Churn Probability Scoring
├── k-Means Behavioral Clustering
└── SQLite Audit Logging (churn_predictions.db)
│
▼
[ Streamlit Dashboard / UI ]


---

## Stack & Components

* **Event Broker:** Apache Kafka & Zookeeper (Dockerized)
* **Inference Worker:** Python consumer script utilizing joblib model artifacts
* **Persistence Layer:** SQLite with Docker bind-mount volume persistence (`- .:/app`)
* **Frontend Visualization:** Streamlit & Plotly

---

## Streaming Module Structure

```text
churn-segmentation-engine/
│
├── consumer.py                    # Kafka consumer, ML inference worker, & SQLite logger
├── stream_producer.py             # Continuous mock customer event generator
├── app_ui.py                      # Streamlit dashboard featuring live SQLite audit logs
├── docker-compose.yml             # Local multi-container stack orchestration
└── README_STREAMING.md            # Streaming extension documentation
```
---

## Running the Streaming Pipeline
1. Prerequisites
Docker and Docker Compose installed on your system.

Python environment with required dependencies installed (pip install -r requirements.txt).

2. Start the Docker Services
Spin up Zookeeper, Kafka, and the inference consumer worker. The volume configuration maps your local project folder to the container, ensuring audit logs persist on your host machine:
```bash
docker-compose up --build
```
(To view real-time container logs in a separate tab, you can run docker-compose logs -f consumer)

3. Start the Event Stream Producer
In a separate terminal window, execute the streaming script to pump live mock customer events into the Kafka broker:
```bash
python stream_producer.py
```

4. Launch the Streamlit Dashboard
In another terminal window, launch the Streamlit app locally to view live audit logs, metrics, and risk distributions read directly from your SQLite database:
```bash
streamlit run app_ui.py
```
---

### **How to Add and Commit Your New File:**

Run these commands in your project terminal to add the streaming documentation to your repository:

```bash
git add README_STREAMING.md
git commit -m "Add real-time streaming and audit pipeline documentation"
git push origin main
```
