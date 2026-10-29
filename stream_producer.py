import json
import random
import time
from kafka import KafkaProducer

# Initialize Kafka Producer connecting to your local Kafka broker
print("[INFO] Connecting Kafka streaming producer to localhost:9092...")
producer = KafkaProducer(
    bootstrap_servers=['localhost:9092'],
    value_serializer=lambda v: json.dumps(v).encode('utf-8')
)

topic_name = 'customer-events'

def generate_random_customer():
    """Generates a realistic mock banking customer profile matching model features."""
    geography = random.choice(["France", "Germany", "Spain"])
    gender = random.choice(["Female", "Male"])
    
    return {
        "CustomerId": random.randint(15000000, 15999999),
        "CreditScore": random.randint(350, 850),
        "Age": random.randint(18, 70),
        "Tenure": random.randint(0, 10),
        "Balance": round(random.uniform(0.0, 250000.0), 2),
        "NumOfProducts": random.randint(1, 4),
        "HasCrCard": random.choice([0, 1]),
        "IsActiveMember": random.choice([0, 1]),
        "EstimatedSalary": round(random.uniform(20000.0, 150000.0), 2),
        "Geography_France": 1 if geography == "France" else 0,
        "Geography_Germany": 1 if geography == "Germany" else 0,
        "Geography_Spain": 1 if geography == "Spain" else 0,
        "Gender_Female": 1 if gender == "Female" else 0,
        "Gender_Male": 1 if gender == "Male" else 0
    }

print(f"[INFO] Starting continuous event stream to topic '{topic_name}' (Press Ctrl+C to stop)...")

try:
    event_count = 0
    while True:
        customer_event = generate_random_customer()
        event_count += 1
        
        # Publish to Kafka
        future = producer.send(topic_name, value=customer_event)
        result = future.get(timeout=5)
        
        print(f"[PRODUCER] Sent Event #{event_count} | Customer ID: {customer_event['CustomerId']} | Partition: {result.partition} | Offset: {result.offset}")
        
        # Wait 2 seconds before streaming the next customer record
        time.sleep(2)

except KeyboardInterrupt:
    print("\n[INFO] Stopping event stream producer...")
    producer.flush()
    producer.close()
    print("[INFO] Producer closed gracefully.")