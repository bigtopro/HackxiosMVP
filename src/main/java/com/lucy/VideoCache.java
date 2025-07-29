package com.lucy;

import com.google.gson.Gson;
import com.google.gson.GsonBuilder;
import com.google.gson.reflect.TypeToken;

import java.io.*;
import java.lang.reflect.Type;
import java.util.Map;
import java.util.concurrent.ConcurrentHashMap;

public class VideoCache {
    private static final String CACHE_FILE = "video_cache.json";
    private final Map<String, CacheEntry> cache;
    private final Gson gson;

    public static class CacheEntry {
        public String videoId;
        public long commentCount;
        public boolean commentsDisabled;
        public long lastUpdated;

        public CacheEntry(String videoId, long commentCount, boolean commentsDisabled) {
            this.videoId = videoId;
            this.commentCount = commentCount;
            this.commentsDisabled = commentsDisabled;
            this.lastUpdated = System.currentTimeMillis();
        }
    }

    public VideoCache() {
        this.gson = new GsonBuilder().setPrettyPrinting().create();
        this.cache = loadCache();
    }

    private Map<String, CacheEntry> loadCache() {
        File file = new File(CACHE_FILE);
        if (!file.exists()) {
            return new ConcurrentHashMap<>();
        }

        try (Reader reader = new FileReader(file)) {
            Type type = new TypeToken<ConcurrentHashMap<String, CacheEntry>>(){}.getType();
            return gson.fromJson(reader, type);
        } catch (IOException e) {
            System.err.println("[WARN] Failed to load cache: " + e.getMessage());
            return new ConcurrentHashMap<>();
        }
    }

    public void saveCache() {
        try (Writer writer = new FileWriter(CACHE_FILE)) {
            gson.toJson(cache, writer);
        } catch (IOException e) {
            System.err.println("[WARN] Failed to save cache: " + e.getMessage());
        }
    }

    public CacheEntry get(String searchQuery) {
        return cache.get(searchQuery);
    }

    public void put(String searchQuery, CacheEntry entry) {
        cache.put(searchQuery, entry);
        // Save after each update to prevent data loss
        saveCache();
    }

    public boolean hasValidEntry(String searchQuery) {
        CacheEntry entry = cache.get(searchQuery);
        if (entry == null) return false;
        
        // Cache entries are valid for 7 days
        long validityPeriod = 7 * 24 * 60 * 60 * 1000L;
        return (System.currentTimeMillis() - entry.lastUpdated) < validityPeriod;
    }
} 