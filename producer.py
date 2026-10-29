import json
from kafka import KafkaProducer

# Initialize Kafka Producer connecting to your local Kafka broker port mapping
print("[INFO] Connecting to Kafka broker at localhost:9092...")
producer = KafkaProducer(
    bootstrap_servers=['localhost:9092'],
    value_serializer=lambda v: json.dumps(v).encode('utf-8')
)

topic_name = 'customer-events'

# Mock customer payload structured to match your model features
mock_customer = {
    "CustomerId": 15634602,
    "CreditScore": 619,
    "Age": 42,
    "Tenure": 2,
    "Balance": 0.0,
    "NumOfProducts": 1,
    "HasCrCard": 1,
    "IsActiveMember": 1,
    "EstimatedSalary": 101348.88,
    "Geography_France": 1,
    "Geography_Germany": 0,
    "Geography_Spain": 0,
    "Gender_Female": 1,
    "Gender_Male": 0
}

print(f"[INFO] Publishing event for Customer {mock_customer['CustomerId']} to Kafka topic '{topic_name}'...")

# Send message to Kafka cluster
future = producer.send(topic_name, value=mock_customer)
result = future.get(timeout=10)

print(f"[SUCCESS] Event published! Partition: {result.partition}, Offset: {result.offset}")

# Ensure delivery and close
producer.flush()
producer.close()