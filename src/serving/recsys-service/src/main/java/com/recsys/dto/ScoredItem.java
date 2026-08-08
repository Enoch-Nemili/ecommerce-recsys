package com.recsys.dto;

public record ScoredItem(long itemId, float rankerScore, double covisitScore) {
}
