"""
Phase 5b — Kafka consumer: updates Redis in real time as events arrive.

This is what makes the system "live." It subscribes to the same topic the
producer (10_kafka_producer.py) publishes to, and for every event that
arrives, immediately updates that user's and item's features in Redis —
using the exact same apply_event_to_redis() logic that was unit-tested
against a real Redis instance in isolation (see streaming_utils.py).

Run this BEFORE starting the producer (or in a separate terminal at the
same time) — a consumer with no messages yet will just wait quietly until
the producer starts sending.

Usage:
    python src/pipeline/11_kafka_consumer.py \
        --topic user-events \
        --redis-host localhost \
        --redis-port 6379
"""

import argparse
import json
import time

import redis
from kafka import KafkaConsumer

from streaming_utils import apply_event_to_redis


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--topic", default="user-events")
    parser.add_argument("--bootstrap-servers", default="localhost:9092")
    parser.add_argument("--group-id", default="recsys-feature-updater")
    parser.add_argument("--redis-host", default="localhost")
    parser.add_argument("--redis-port", type=int, default=6379)
    args = parser.parse_args()

    r = redis.Redis(host=args.redis_host, port=args.redis_port, decode_responses=True)
    r.ping()
    print(f"Connected to Redis at {args.redis_host}:{args.redis_port}")

    consumer = KafkaConsumer(
        args.topic,
        bootstrap_servers=args.bootstrap_servers,
        group_id=args.group_id,
        value_deserializer=lambda v: json.loads(v.decode("utf-8")),
        auto_offset_reset="latest",  # only process events sent AFTER this consumer starts
    )
    print(f"Subscribed to topic '{args.topic}'. Waiting for events ... (Ctrl+C to stop)")

    processed = 0
    start = time.time()
    try:
        for message in consumer:
            event = message.value
            apply_event_to_redis(r, event)
            processed += 1
            if processed % 50 == 0:
                elapsed = time.time() - start
                rate = processed / elapsed if elapsed > 0 else 0
                print(f"  processed {processed:,} events ({rate:.1f}/sec) — "
                      f"last: user={event['user_id']} item={event['item_id']} "
                      f"behavior={event['behavior_type']}")
    except KeyboardInterrupt:
        print(f"\nStopped. Processed {processed:,} events total.")


if __name__ == "__main__":
    main()
