package com.lucy;

import com.google.gson.Gson;
import com.google.gson.GsonBuilder;
import java.io.*;
import java.nio.charset.StandardCharsets;
import java.util.*;

public class SampleCommentsAggregator {
    public static void main(String[] args) throws IOException {
        File commentsDir = new File("comments");
        if (!commentsDir.exists() || !commentsDir.isDirectory()) {
            System.err.println("[ERROR] 'comments' directory not found.");
            return;
        }
        File[] jsonFiles = commentsDir.listFiles((dir, name) -> name.endsWith(".json"));
        if (jsonFiles == null || jsonFiles.length == 0) {
            System.err.println("[INFO] No .json files found in 'comments' directory.");
            return;
        }
        Gson gson = new GsonBuilder().setPrettyPrinting().create();
        List<String> aggregated = new ArrayList<>();
        Random random = new Random();
        for (File jsonFile : jsonFiles) {
            // Skip files that don't follow the videoId_comments.json pattern
            if (!jsonFile.getName().endsWith("_comments.json")) continue;

            List<String> comments;
            try (Reader reader = new InputStreamReader(new FileInputStream(jsonFile), StandardCharsets.UTF_8)) {
                comments = gson.fromJson(reader, List.class);
            }
            if (comments == null || comments.isEmpty()) continue;
            int sampleSize = Math.max(1, (int) Math.round(comments.size() * 0.01));
            // Shuffle and pick first sampleSize
            Collections.shuffle(comments, random);
            aggregated.addAll(comments.subList(0, Math.min(sampleSize, comments.size())));
            System.out.println("Sampled " + sampleSize + " from " + jsonFile.getName());
        }
        // Write aggregated to a single JSON file
        File output = new File("sampled_comments.json");
        try (Writer writer = new OutputStreamWriter(new FileOutputStream(output), StandardCharsets.UTF_8)) {
            gson.toJson(aggregated, writer);
        }
        System.out.println("Aggregated " + aggregated.size() + " sampled comments into 'sampled_comments.json'.");
    }
} 