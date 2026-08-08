package com.recsys.controller;

import ai.onnxruntime.OrtException;
import com.recsys.dto.RecommendationResponse;
import com.recsys.dto.ScoredItem;
import com.recsys.service.RankerService;
import com.recsys.service.RedisService;
import org.springframework.web.bind.annotation.GetMapping;
import org.springframework.web.bind.annotation.RequestParam;
import org.springframework.web.bind.annotation.RestController;

import java.util.List;
import java.util.Map;
import java.util.Comparator;

/**
 * The online serving path: given a user, this pulls their precomputed
 * candidates, fetches features, scores each candidate with the ranker,
 * and returns the top N sorted by score.
 *
 * Deliberately does NOT do any heavy computation here — candidate
 * generation already happened offline (06_build_user_candidates.py), and
 * feature lookups are single-hop Redis reads. This is what keeps a real
 * recommendation request fast: everything expensive was already done in
 * batch, ahead of time.
 */
@RestController
public class RecommendController {

    private final RedisService redisService;
    private final RankerService rankerService;

    public RecommendController(RedisService redisService, RankerService rankerService) {
        this.redisService = redisService;
        this.rankerService = rankerService;
    }

    @GetMapping("/recommend")
    public RecommendationResponse recommend(
            @RequestParam("user") long userId,
            @RequestParam(value = "topN", defaultValue = "10") int topN,
            @RequestParam(value = "candidatePoolSize", defaultValue = "50") int candidatePoolSize
    ) throws OrtException {

        Map<Long, Double> candidatesWithCovisitScore = redisService.getCandidateItemsWithScores(userId, candidatePoolSize);
        if (candidatesWithCovisitScore.isEmpty()) {
            // No precomputed candidates for this user (e.g. no history in
            // the offline window). Real systems fall back to a popularity
            // list here; that fallback is a good next enhancement but out
            // of scope for this first version — return empty rather than
            // silently guessing.
            return new RecommendationResponse(userId, List.of());
        }

        List<Long> candidateItemIds = candidatesWithCovisitScore.keySet().stream().toList();
        Map<String, String> userFeatures = redisService.getUserFeatures(userId);
        Map<Long, Map<String, String>> itemFeaturesById = redisService.getItemFeaturesBatch(candidateItemIds);

        List<ScoredItem> scored = candidateItemIds.stream()
                .map(itemId -> {
                    Map<String, String> itemFeatures = itemFeaturesById.getOrDefault(itemId, Map.of());
                    float rankerScore;
                    try {
                        rankerScore = rankerService.score(userFeatures, itemFeatures);
                    } catch (OrtException e) {
                        rankerScore = 0f; // scoring failure for one item shouldn't fail the whole request
                    }
                    double covisitScore = candidatesWithCovisitScore.get(itemId);
                    return new ScoredItem(itemId, rankerScore, covisitScore);
                })
                // Primary sort: the ranker's score. Secondary sort (tiebreaker):
                // co-visitation strength. The ranker can saturate near 1.0 for
                // any "clearly good" candidate when positives are rare in
                // training (which they are here — ~5.7% positive rate), so on
                // its own it can't always distinguish "great match" from
                // "slightly better match." Co-visitation strength — how
                // strongly this item is linked to what the user already
                // engaged with — is a reasonable secondary signal for exactly
                // that situation.
                .sorted(
                        Comparator.comparingDouble(ScoredItem::rankerScore)
                                .thenComparingDouble(ScoredItem::covisitScore)
                                .reversed()
                )
                .limit(topN)
                .toList();

        return new RecommendationResponse(userId, scored);
    }
}
