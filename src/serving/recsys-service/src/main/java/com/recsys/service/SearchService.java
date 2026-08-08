package com.recsys.service;

import com.fasterxml.jackson.databind.JsonNode;
import com.fasterxml.jackson.databind.ObjectMapper;
import org.springframework.beans.factory.annotation.Value;
import org.springframework.stereotype.Service;

import java.io.IOException;
import java.net.URI;
import java.net.http.HttpClient;
import java.net.http.HttpRequest;
import java.net.http.HttpResponse;
import java.util.ArrayList;
import java.util.List;

/**
 * Talks to Elasticsearch directly over its REST API (POST .../_search with
 * a JSON query body) rather than pulling in the official Elasticsearch
 * Java client library. This keeps the dependency footprint small and low-
 * risk — java.net.http.HttpClient and Jackson's ObjectMapper are both
 * already used elsewhere in this project (Spring's web starter brings in
 * Jackson), so nothing new and unproven is being introduced here.
 */
@Service
public class SearchService {

    private final HttpClient httpClient;
    private final ObjectMapper mapper;
    private final String esBaseUrl;
    private final String index;

    public SearchService(
            @Value("${elasticsearch.host:http://localhost:9200}") String esBaseUrl,
            @Value("${elasticsearch.index:items}") String index) {
        this.httpClient = HttpClient.newHttpClient();
        this.mapper = new ObjectMapper();
        this.esBaseUrl = esBaseUrl;
        this.index = index;
    }

    /**
     * Full-text search against the (synthetic) title field, optionally
     * filtered to a category and boosted by popularity_score — this
     * combined query is closer to how a real product search page ranks
     * results than pure text relevance alone would be.
     */
    public List<SearchResultItem> search(String queryText, Long categoryId, int topN) throws IOException, InterruptedException {
        ObjectMapper m = mapper;
        var root = m.createObjectNode();
        var functionScore = root.putObject("query").putObject("function_score");

        var boolQuery = functionScore.putObject("query").putObject("bool");

        // Built with explicit intermediate variables rather than a single
        // chained expression — chaining .putObject(...) (which returns the
        // newly created INNER object) together with .put(...) (which
        // returns the object it was called ON, i.e. `this`) silently picks
        // up the wrong node as the chain's final value, dropping the outer
        // wrapper. Assigning each level to its own variable makes it
        // unambiguous which node is being built and returned.
        var matchWrapper = m.createObjectNode();
        var titleMatch = matchWrapper.putObject("match").putObject("title");
        titleMatch.put("query", queryText);
        titleMatch.put("fuzziness", "AUTO");
        boolQuery.putArray("must").add(matchWrapper);

        if (categoryId != null) {
            var termWrapper = m.createObjectNode();
            termWrapper.putObject("term").put("category_id", categoryId);
            boolQuery.putArray("filter").add(termWrapper);
        }

        var functionWrapper = m.createObjectNode();
        var fieldValueFactor = functionWrapper.putObject("field_value_factor");
        fieldValueFactor.put("field", "popularity_score");
        fieldValueFactor.put("modifier", "log1p");
        fieldValueFactor.put("missing", 0);
        functionScore.putArray("functions").add(functionWrapper);
        functionScore.put("boost_mode", "sum");

        root.put("size", topN);

        String requestBody = m.writeValueAsString(root);

        HttpRequest request = HttpRequest.newBuilder()
                .uri(URI.create(esBaseUrl + "/" + index + "/_search"))
                .header("Content-Type", "application/json")
                .POST(HttpRequest.BodyPublishers.ofString(requestBody))
                .build();

        HttpResponse<String> response = httpClient.send(request, HttpResponse.BodyHandlers.ofString());

        List<SearchResultItem> results = new ArrayList<>();
        if (response.statusCode() != 200) {
            // Surface the ES error body rather than swallowing it — makes
            // a misconfigured index or bad query visible instead of just
            // silently returning an empty list.
            throw new IOException("Elasticsearch returned " + response.statusCode() + ": " + response.body());
        }

        JsonNode hits = mapper.readTree(response.body()).path("hits").path("hits");
        for (JsonNode hit : hits) {
            JsonNode src = hit.path("_source");
            results.add(new SearchResultItem(
                    src.path("item_id").asLong(),
                    src.path("title").asText(),
                    src.hasNonNull("category_id") ? src.path("category_id").asLong() : null,
                    src.path("popularity_score").asDouble(),
                    (float) hit.path("_score").asDouble()
            ));
        }
        return results;
    }

    public record SearchResultItem(long itemId, String title, Long categoryId, double popularityScore, float score) {
    }
}
