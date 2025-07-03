package com.lucy;

import java.util.ArrayList;
import java.util.List;

public class ApiKeyManager {
    private final List<String> apiKeys;
    private final boolean[] exhausted;
    private int currentIndex = 0;

    public ApiKeyManager(String[] keys) {
        this.apiKeys = new ArrayList<>();
        for (String key : keys) {
            apiKeys.add(key);
        }
        this.exhausted = new boolean[apiKeys.size()];
    }

    public synchronized String getCurrentKey() {
        return apiKeys.get(currentIndex);
    }

    public synchronized void markCurrentKeyExhausted() {
        exhausted[currentIndex] = true;
    }

    public synchronized boolean allKeysExhausted() {
        for (boolean b : exhausted) {
            if (!b) return false;
        }
        return true;
    }

    public synchronized boolean switchToNextKey() {
        int start = currentIndex;
        do {
            currentIndex = (currentIndex + 1) % apiKeys.size();
            if (!exhausted[currentIndex]) {
                return true;
            }
        } while (currentIndex != start);
        return false; // All keys exhausted
    }

    public synchronized void resetAllKeys() {
        for (int i = 0; i < exhausted.length; i++) exhausted[i] = false;
        currentIndex = 0;
    }
} 