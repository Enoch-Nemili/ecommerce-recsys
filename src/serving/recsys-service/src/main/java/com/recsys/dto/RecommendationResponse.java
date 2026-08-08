package com.recsys.dto;

import java.util.List;

public record RecommendationResponse(long userId, List<ScoredItem> recommendations) {
}
