package com.recsys.service;

import org.springframework.beans.factory.annotation.Value;
import org.springframework.stereotype.Service;
import redis.clients.jedis.Jedis;
import redis.clients.jedis.JedisPool;
import redis.clients.jedis.resps.Tuple;

import java.util.LinkedHashMap;
import java.util.List;
import java.util.Map;

/**
 * Wraps all Redis lookups the service needs. Feature data was loaded
 * offline by 04_load_redis.py (item:{id}, user:{id} hashes) and
 * 07_load_candidates_redis.py (cands:{id} sorted sets) — this class only
 * reads, it never writes, keeping the online path simple and fast.
 */
@Service
public class RedisService {

    private final JedisPool jedisPool;

    public RedisService(
            @Value("${redis.host:localhost}") String host,
            @Value("${redis.port:6379}") int port) {
        this.jedisPool = new JedisPool(host, port);
    }

    /** Fetches a user's precomputed candidate items, already ranked by
     * co-visitation score, highest first. Empty list if the user has no
     * precomputed candidates (e.g. a brand-new user with no history). */
    public List<Long> getCandidateItems(long userId, int limit) {
        try (Jedis jedis = jedisPool.getResource()) {
            List<String> ids = jedis.zrevrange("cands:" + userId, 0, limit - 1);
            return ids.stream().map(Long::parseLong).toList();
        }
    }

    /** Same as getCandidateItems, but keeps the co-visitation score for
     * each item. Used as a secondary sort key: the ranker model can
     * saturate (output near-identical scores) for items that are all
     * "clearly good enough," and co-visitation strength is a reasonable
     * tiebreaker in that case — items more strongly linked to what the
     * user already engaged with rank above weaker matches. */
    public Map<Long, Double> getCandidateItemsWithScores(long userId, int limit) {
        try (Jedis jedis = jedisPool.getResource()) {
            List<Tuple> tuples = jedis.zrevrangeWithScores("cands:" + userId, 0, limit - 1);
            Map<Long, Double> result = new LinkedHashMap<>();
            for (Tuple t : tuples) {
                result.put(Long.parseLong(t.getElement()), t.getScore());
            }
            return result;
        }
    }

    /** Raw string->string feature map for a user. Empty map if not found. */
    public Map<String, String> getUserFeatures(long userId) {
        try (Jedis jedis = jedisPool.getResource()) {
            return jedis.hgetAll("user:" + userId);
        }
    }

    /** Raw string->string feature map for an item. Empty map if not found. */
    public Map<String, String> getItemFeatures(long itemId) {
        try (Jedis jedis = jedisPool.getResource()) {
            return jedis.hgetAll("item:" + itemId);
        }
    }

    /** Batched item feature fetch — one round trip per item is fine at the
     * candidate-list scale we're working with (tens of items), but this
     * keeps the door open for pipelining if that scale grows later. */
    public Map<Long, Map<String, String>> getItemFeaturesBatch(List<Long> itemIds) {
        Map<Long, Map<String, String>> result = new LinkedHashMap<>();
        try (Jedis jedis = jedisPool.getResource()) {
            for (Long itemId : itemIds) {
                result.put(itemId, jedis.hgetAll("item:" + itemId));
            }
        }
        return result;
    }
}
