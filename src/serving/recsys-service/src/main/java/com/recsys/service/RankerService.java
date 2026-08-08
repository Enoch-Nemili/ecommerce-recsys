package com.recsys.service;

import ai.onnxruntime.*;
import jakarta.annotation.PreDestroy;
import org.springframework.beans.factory.annotation.Value;
import org.springframework.stereotype.Service;

import java.util.HashMap;
import java.util.Map;

/**
 * Loads the ONNX-exported LightGBM ranker once at startup and scores
 * (user, item) feature vectors on demand. Feature order here MUST match
 * FEATURE_COLS in 03_train_ranker.py / 05_export_onnx.py exactly — the
 * ONNX model has no column names, only positions.
 */
@Service
public class RankerService {

    private static final String[] FEATURE_ORDER = {
            "total_events", "num_sessions", "buy_count",           // user features
            "item_total_events", "item_buy_count", "view_count",
            "distinct_users", "popularity_score",                   // item features
    };

    private final OrtEnvironment env;
    private final OrtSession session;

    public RankerService(@Value("${ranker.model.path}") String modelPath) throws OrtException {
        this.env = OrtEnvironment.getEnvironment();
        OrtSession.SessionOptions opts = new OrtSession.SessionOptions();
        this.session = env.createSession(modelPath, opts);
    }

    /**
     * Scores one (user, item) pair. userFeatures and itemFeatures are the
     * raw string->string maps straight from Redis (HGETALL) — missing
     * fields default to 0, so a partially-known user/item still gets a
     * (lower-confidence) score instead of failing the request.
     */
    public float score(Map<String, String> userFeatures, Map<String, String> itemFeatures) throws OrtException {
        Map<String, String> merged = new HashMap<>(userFeatures);
        // item_total_events / item_buy_count are the item table's total_events/
        // buy_count columns, renamed to avoid clashing with the user's own
        // total_events/buy_count — matches the renaming done in
        // 02_build_training_examples.py.
        merged.put("item_total_events", itemFeatures.getOrDefault("total_events", "0"));
        merged.put("item_buy_count", itemFeatures.getOrDefault("buy_count", "0"));
        merged.put("view_count", itemFeatures.getOrDefault("view_count", "0"));
        merged.put("distinct_users", itemFeatures.getOrDefault("distinct_users", "0"));
        merged.put("popularity_score", itemFeatures.getOrDefault("popularity_score", "0"));

        float[] row = new float[FEATURE_ORDER.length];
        for (int i = 0; i < FEATURE_ORDER.length; i++) {
            row[i] = parseFloatSafe(merged.get(FEATURE_ORDER[i]));
        }

        try (OnnxTensor tensor = OnnxTensor.createTensor(env, new float[][]{row})) {
            Map<String, OnnxTensor> feeds = new HashMap<>();
            feeds.put("input", tensor);
            try (OrtSession.Result result = session.run(feeds)) {
                float[][] probs = (float[][]) result.get("probabilities").get().getValue();
                return probs[0][1]; // probability of the positive class = ranking score
            }
        }
    }

    private static float parseFloatSafe(String s) {
        if (s == null) return 0f;
        try {
            return Float.parseFloat(s);
        } catch (NumberFormatException e) {
            return 0f;
        }
    }

    @PreDestroy
    public void close() {
        try {
            if (session != null) session.close();
            if (env != null) env.close();
        } catch (OrtException e) {
            // Best-effort cleanup on shutdown — nothing meaningful to do
            // with this exception, and rethrowing would just interfere
            // with the shutdown sequence.
        }
    }
}
