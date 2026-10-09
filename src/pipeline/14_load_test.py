"""
Phase 7a — Load test the /recommend and /search endpoints.

Simulates concurrent users hitting the live Java service, and reports
real latency percentiles (p50/p95/p99) and throughput (requests/sec) —
the actual numbers that back up a "high-performance service" claim,
instead of just asserting it.

A realistic user distribution matters here: real traffic to a
recommendation system is overwhelmingly /recommend (loading a page,
scrolling a feed) with /search being comparatively rare (someone actively
typing a query). @task weights below reflect that — recommend is weighted
5x more likely per simulated user than search — so the aggregate numbers
resemble a real traffic mix instead of testing both endpoints equally,
which would understate how the system performs under its actual expected
load shape.

Usage (headless, prints a summary and exits):
    locust -f src/pipeline/14_load_test.py \
        --host http://localhost:8080 \
        --users 50 --spawn-rate 10 --run-time 60s --headless

Or run with the web UI instead (omit --headless) and open
http://localhost:8089 to watch it live and control ramp-up interactively.
"""

import random

from locust import HttpUser, between, task

# A spread of real user_ids and category_ids from the actual dataset —
# using a range rather than one fixed ID exercises Redis/ES with varied
# keys instead of hammering (and artificially caching-favoring) a single
# hot key the whole run.
SAMPLE_USER_IDS = list(range(1, 5000))
SAMPLE_CATEGORY_IDS = [2885642, 4756105, 3607361, 2735466, 4129924]
SAMPLE_SEARCH_TERMS = ["Item 1", "Item 100", "Item 500", "Item 1000", "Item 5000"]


class RecsysUser(HttpUser):
    # Real users don't fire requests back-to-back with zero gap — this
    # models a small think-time between actions, closer to actual browsing
    # behavior than a tight request loop would be.
    wait_time = between(0.1, 1.0)

    @task(5)
    def get_recommendations(self):
        user_id = random.choice(SAMPLE_USER_IDS)
        self.client.get(
            f"/recommend?user={user_id}&topN=10",
            name="/recommend",
        )

    @task(1)
    def search(self):
        term = random.choice(SAMPLE_SEARCH_TERMS)
        category = random.choice(SAMPLE_CATEGORY_IDS)
        self.client.get(
            f"/search?q={term}&category={category}&topN=10",
            name="/search",
        )
