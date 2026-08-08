"""
Phase 5a — Kafka producer: replays historical events as a live stream.

In a real system, events would arrive on Kafka the moment a user does
something on the site. We don't have a live website, so this script
replays real historical events from UserBehavior.csv onto a Kafka topic,
optionally paced (--rate) to simulate them arriving over time rather than
all at once — useful for actually watching the consumer update Redis live
during a demo, instead of it all happening faster than you can observe.

Usage:
    python src/pipeline/10_kafka_producer.py \
        --input data/raw/UserBehavior.csv \
        --topic user-events \
        --limit 2000 \
        --rate 50
"""

import argparse
import csv
import json
import time

from kafka import KafkaProducer

SCHEMA_COLS = ["user_id", "item_id", "category_id", "behavior_type", "ts"]


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--input", required=True)
    parser.add_argument("--topic", default="user-events")
    parser.add_argument("--bootstrap-servers", default="localhost:9092")
    parser.add_argument("--limit", type=int, default=2000, help="Max events to replay")
    parser.add_argument("--rate", type=float, default=50.0, help="Events per second (paced replay)")
    parser.add_argument("--user-filter", type=int, default=None,
                         help="Only replay events for this user_id (useful for a focused demo)")
    args = parser.parse_args()

    producer = KafkaProducer(
        bootstrap_servers=args.bootstrap_servers,
        value_serializer=lambda v: json.dumps(v).encode("utf-8"),
    )
    print(f"Connected to Kafka at {args.bootstrap_servers}, publishing to topic '{args.topic}'")

    delay = 1.0 / args.rate if args.rate > 0 else 0
    sent = 0

    with open(args.input, newline="") as f:
        reader = csv.reader(f)
        for row in reader:
            if sent >= args.limit:
                break
            event = dict(zip(SCHEMA_COLS, row))
            event["user_id"] = int(event["user_id"])
            event["item_id"] = int(event["item_id"])
            event["ts"] = int(event["ts"])

            if args.user_filter is not None and event["user_id"] != args.user_filter:
                continue

            producer.send(args.topic, value=event)
            sent += 1
            if sent % 100 == 0:
                print(f"  sent {sent:,} events ...")
            if delay > 0:
                time.sleep(delay)

    producer.flush()
    print(f"Done. Sent {sent:,} events to topic '{args.topic}'.")


if __name__ == "__main__":
    main()
