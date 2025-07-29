package com.lucy;

import com.google.gson.Gson;
import com.google.gson.GsonBuilder;
import com.google.gson.reflect.TypeToken;

import java.io.*;
import java.util.Map;
import java.util.concurrent.ConcurrentHashMap;
import java.util.concurrent.atomic.AtomicInteger;

public class ApiKeyManager {
    private static final String EXHAUSTED_KEYS_FILE = "exhausted_keys.json";
    private static final long KEY_COOLDOWN_MILLIS = 24 * 60 * 60 * 1000L; // 24 hours

    private final String[] apiKeys;
    private final Map<String, Long> exhaustedKeyTimestamps;
    private final AtomicInteger currentKeyIndex = new AtomicInteger(0);

    public ApiKeyManager(String[] apiKeys) {
        this.apiKeys = apiKeys;
        this.exhaustedKeyTimestamps = new ConcurrentHashMap<>();
        loadExhaustedKeys();
    }

    private void loadExhaustedKeys() {
        File file = new File(EXHAUSTED_KEYS_FILE);
        if (!file.exists()) {
            return;
        }
        try (Reader reader = new FileReader(file)) {
            Map<String, Long> storedTimestamps = new Gson().fromJson(reader, new TypeToken<Map<String, Long>>() {}.getType());
            if (storedTimestamps != null) {
                exhaustedKeyTimestamps.putAll(storedTimestamps);
            }
        } catch (Exception e) {
            System.err.println("[WARN] Failed to load exhausted keys file: " + e.getMessage());
        }
        // Prune keys that are no longer in cooldown
        pruneExpiredKeys();
    }

    private synchronized void saveExhaustedKeys() {
        try (Writer writer = new FileWriter(EXHAUSTED_KEYS_FILE)) {
            new GsonBuilder().setPrettyPrinting().create().toJson(exhaustedKeyTimestamps, writer);
        } catch (IOException e) {
            System.err.println("[WARN] Failed to save exhausted keys file: " + e.getMessage());
        }
    }
    
    private void pruneExpiredKeys() {
        long now = System.currentTimeMillis();
        if (exhaustedKeyTimestamps.entrySet().removeIf(entry -> (now - entry.getValue() >= KEY_COOLDOWN_MILLIS))) {
            System.out.println("[INFO] One or more API keys have completed their 24h cooldown and are now available.");
            saveExhaustedKeys();
        }
    }

    public synchronized String getNextAvailableKey() throws InterruptedException {
        pruneExpiredKeys(); // Always check for expired keys before getting a new one.

        int startIndex = currentKeyIndex.get();
        int totalKeys = apiKeys.length;

        for (int i = 0; i < totalKeys; i++) {
            String key = apiKeys[currentKeyIndex.get()];
            if (!exhaustedKeyTimestamps.containsKey(key)) {
                // Move to next key for next time to distribute load
                currentKeyIndex.set((currentKeyIndex.get() + 1) % totalKeys);
                return key;
            }
            // Move to the next key
            currentKeyIndex.set((currentKeyIndex.get() + 1) % totalKeys);
        }

        // If we've looped through all keys and found none, wait.
        System.err.println("[ERROR] All API keys are currently exhausted. Waiting for 5 minutes before retrying...");
        Thread.sleep(5 * 60 * 1000); // Wait 5 minutes
        return getNextAvailableKey(); // Retry getting a key
    }

    public synchronized void recordQuotaExceeded(String apiKey) {
        if (exhaustedKeyTimestamps.containsKey(apiKey)) return; // Already marked
        System.out.printf("[WARN] API key %s... has hit its quota. Placing it in 24-hour cooldown.%n", apiKey.substring(0, 10));
        exhaustedKeyTimestamps.put(apiKey, System.currentTimeMillis());
        saveExhaustedKeys();
    }
} 