package com.recsys.controller;

import com.recsys.dto.SearchResponse;
import com.recsys.service.SearchService;
import org.springframework.web.bind.annotation.GetMapping;
import org.springframework.web.bind.annotation.RequestParam;
import org.springframework.web.bind.annotation.RestController;

import java.io.IOException;

/**
 * GET /search?q=...&category=...&topN=...
 *
 * Product search, separate from /recommend. Where /recommend answers
 * "what should we show this user," /search answers "the user typed
 * something, find matching items" — a genuinely different use case, and
 * the one Elasticsearch is purpose-built for.
 */
@RestController
public class SearchController {

    private final SearchService searchService;

    public SearchController(SearchService searchService) {
        this.searchService = searchService;
    }

    @GetMapping("/search")
    public SearchResponse search(
            @RequestParam("q") String query,
            @RequestParam(value = "category", required = false) Long categoryId,
            @RequestParam(value = "topN", defaultValue = "10") int topN
    ) throws IOException, InterruptedException {
        var results = searchService.search(query, categoryId, topN);
        return new SearchResponse(query, results);
    }
}
