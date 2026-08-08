"""
Shared real-time update logic, used by 11_kafka_consumer.py.

Kept in its own module (rather than inline in the consumer) so this exact
function can be unit-tested against a real Redis instance independently of
whether Kafka itself is running — this is the part that actually matters:
Kafka just delivers the message, this is what makes the system "live."

Mirrors the same BEHAVIOR_WEIGHT scheme used in 01_sessionize_and_candidates.py
so real-time popularity scores stay consistent with the offline batch
computation, rather than drifting into a different scale over time.
"""

BEHAVIOR_WEIGHT = {"pv": 1.0, "fav": 2.0, "cart": 3.0, "buy": 5.0}


def apply_event_to_redis(r, event: dict) -> None:
    """Applies one (user_id, item_id, category_id, behavior_type, ts) event
    to the live Redis feature store, incrementally — no Spark job, no
    waiting for the next batch run. This is intentionally a small, focused
    set of updates:

    User side: bump total_events, bump buy_count on a purchase, and stamp
    last_active with this event's timestamp (events are assumed to arrive
    in roughly chronological order from the producer, so a plain overwrite
    is fine here — a production system replaying out-of-order events would
    need a compare-and-set instead).

    Item side: bump total_events / buy_count / view_count as applicable,
    track distinct viewers via a Redis SET (so distinct_users stays
    accurate rather than becoming an ever-growing counter that never
    reflects "distinct"), and bump popularity_score by this event's weight.
    """
    user_id = event["user_id"]
    item_id = event["item_id"]
    behavior_type = event["behavior_type"]
    ts = event["ts"]
    weight = BEHAVIOR_WEIGHT.get(behavior_type, 0.0)

    user_key = f"user:{user_id}"
    pipe = r.pipeline(transaction=False)
    pipe.hincrby(user_key, "total_events", 1)
    if behavior_type == "buy":
        pipe.hincrby(user_key, "buy_count", 1)
    pipe.hset(user_key, "last_active_ts", ts)

    item_key = f"item:{item_id}"
    pipe.hincrby(item_key, "total_events", 1)
    if behavior_type == "buy":
        pipe.hincrby(item_key, "buy_count", 1)
    if behavior_type == "pv":
        pipe.hincrby(item_key, "view_count", 1)
    pipe.hincrbyfloat(item_key, "popularity_score", weight)

    # Distinct-viewer tracking: a Redis SET of user IDs per item. SCARD
    # gives the true distinct count on read — an ordinary counter can't do
    # this correctly, since it would double-count repeat visits from the
    # same user.
    item_users_key = f"item:{item_id}:users"
    pipe.sadd(item_users_key, user_id)

    pipe.execute()
