package com.recsys.dto;

import com.recsys.service.SearchService;

import java.util.List;

public record SearchResponse(String query, List<SearchService.SearchResultItem> results) {
}
