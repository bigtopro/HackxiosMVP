package com.lucy;

import com.google.api.client.googleapis.javanet.GoogleNetHttpTransport;
import com.google.api.client.http.javanet.NetHttpTransport;
import com.google.api.client.json.JsonFactory;
import com.google.api.client.json.gson.GsonFactory;
import com.google.api.services.youtube.YouTube;
import com.google.api.services.youtube.model.CommentThread;
import com.google.api.services.youtube.model.CommentThreadListResponse;
import com.google.gson.Gson;
import com.google.gson.GsonBuilder;
import com.google.gson.reflect.TypeToken;

import java.io.*;
import java.nio.charset.StandardCharsets;
import java.security.GeneralSecurityException;
import java.util.*;
import java.util.concurrent.*;
import com.google.api.client.googleapis.json.GoogleJsonResponseException;

import java.nio.file.Files;
import java.nio.file.Paths;
import java.time.Instant;
import java.time.Duration;
import com.google.gson.JsonObject;
import com.google.gson.JsonParser;
import java.util.concurrent.ConcurrentHashMap;
import java.util.concurrent.atomic.AtomicBoolean;
import java.util.concurrent.atomic.AtomicInteger;

public class YoutubeCommentScraper {

    private static final String APPLICATION_NAME = "YouTube-Comment-Fetcher";
    private static final JsonFactory JSON_FACTORY = GsonFactory.getDefaultInstance();
    private static final int COMMENTS_PER_PAGE = 100;

    public static final String[] API_KEYS = loadApiKeysFromEnv();

    private static String[] loadApiKeysFromEnv() {
        List<String> keys = new ArrayList<>();

        // Try to load up to 20 API keys from environment variables
        for (int i = 1; i <= 20; i++) {
            String envKeyName = "YOUTUBE_API_KEY_" + i;
            String key = System.getenv(envKeyName);

            // If not found in environment, try to load from .env file
            if (key == null || key.trim().isEmpty()) {
                key = loadFromEnvFile(envKeyName);
            }

            if (key != null && !key.trim().isEmpty()) {
                keys.add(key.trim());
            }
        }

        if (keys.isEmpty()) {
            System.err.println("[ERROR] No YouTube API keys found in environment variables!");
            System.err.println("Please set YOUTUBE_API_KEY_1, YOUTUBE_API_KEY_2, etc. in your environment or .env file");
        } else {
            System.out.println("[INFO] Loaded " + keys.size() + " YouTube API keys from environment");
        }

        return keys.toArray(new String[0]);
    }

    private static String loadFromEnvFile(String keyName) {
        try {
            java.nio.file.Path envPath = java.nio.file.Paths.get(".env");
            if (java.nio.file.Files.exists(envPath)) {
                java.util.List<String> lines = java.nio.file.Files.readAllLines(envPath);
                for (String line : lines) {
                    if (line.trim().startsWith(keyName + "=")) {
                        String value = line.split("=", 2)[1].trim();
                        if (value.startsWith("\"") && value.endsWith("\"")) {
                            value = value.substring(1, value.length() - 1);
                        }
                        if (value.startsWith("'") && value.endsWith("'")) {
                            value = value.substring(1, value.length() - 1);
                        }
                        return value;
                    }
                }
            }
        } catch (Exception e) {
            System.err.println("[WARN] Could not read .env file: " + e.getMessage());
        }
        return null;
    }

    private static class CommentPage {
        int pageNumber;
        String pageToken;
        List<String> comments;

        CommentPage(int pageNumber, String pageToken, List<String> comments) {
            this.pageNumber = pageNumber;
            this.pageToken = pageToken;
            this.comments = comments;
        }
    }

    private static List<String> fetchAllReplies(CommentThread commentThread, ApiKeyManager apiKeyManager, YouTube youtubeService) throws InterruptedException {
        List<String> replies = new ArrayList<>();
        String parentId = commentThread.getSnippet().getTopLevelComment().getId();
        String pageToken = null;
        int maxRetries = 3;

        while (true) {
            String currentKey = apiKeyManager.getNextAvailableKey();
            int retries = 0;

            try {
                YouTube.Comments.List request = youtubeService.comments()
                        .list("snippet")
                        .setParentId(parentId)
                        .setMaxResults(100L)
                        .setTextFormat("plainText")
                        .setKey(currentKey);

                if (pageToken != null) {
                    request.setPageToken(pageToken);
                }

                com.google.api.services.youtube.model.CommentListResponse response = request.execute();

                for (com.google.api.services.youtube.model.Comment reply : response.getItems()) {
                    String comment = reply.getSnippet().getTextDisplay();
                    String cleaned = comment.replaceAll("https?://\\S+", "").trim().replaceAll("[\n\r]+", " ");
                    if (!cleaned.isEmpty()) {
                        replies.add(cleaned);
                    }
                }

                pageToken = response.getNextPageToken();
                if (pageToken == null) {
                    break;
                }
                Thread.sleep(50); // Be nice to the API
                retries = 0; // Reset retries on success

            } catch (GoogleJsonResponseException e) {
                if (e.getStatusCode() == 403 && e.getMessage() != null && e.getMessage().contains("quota")) {
                    apiKeyManager.recordQuotaExceeded(currentKey);
                    // Let the while loop try again with a new key
                } else {
                    retries++;
                     if (retries >= maxRetries) {
                        System.err.printf("[ERROR] Exceeded max retries fetching replies for comment %s. Skipping. Error: %s%n", parentId, e.getMessage());
                        break;
                    }
                    System.err.printf("[WARNING] Retrying to fetch replies for comment %s (%d/%d)...%n", parentId, retries, maxRetries);
                    Thread.sleep(1000 * retries); // backoff
                }
            } catch (IOException e) {
                 retries++;
                 if (retries >= maxRetries) {
                    System.err.printf("[ERROR] Exceeded max retries fetching replies for comment %s due to IOException. Skipping. Error: %s%n", parentId, e.getMessage());
                    break;
                 }
                 System.err.printf("[WARNING] Retrying to fetch replies for comment %s (%d/%d) due to IOException...%n", parentId, retries, maxRetries);
                 Thread.sleep(1000 * retries); // backoff
            }
        }
        return replies;
    }

    public static void fetchCommentsParallel(String videoId, String outputFile, int numThreads, ApiKeyManager apiKeyManager) throws Exception {
        System.out.println("[INFO] Pre-fetching page tokens for video: " + videoId);
        List<String> allPageTokens = new ArrayList<>();
        String pageToken = null;
        YouTube youtubeService = new YouTube.Builder(
                GoogleNetHttpTransport.newTrustedTransport(), JSON_FACTORY, null)
                .setApplicationName(APPLICATION_NAME)
                .build();

        // Always include first page (token "" represents null)
        allPageTokens.add("");
        while (true) {
            String currentKey = apiKeyManager.getNextAvailableKey();
            try {
                YouTube.CommentThreads.List request = youtubeService.commentThreads()
                        .list("snippet")
                        .setVideoId(videoId)
                        .setMaxResults((long) COMMENTS_PER_PAGE)
                        .setTextFormat("plainText")
                        .setKey(currentKey);

                if (pageToken != null) {
                    request.setPageToken(pageToken);
                }

                CommentThreadListResponse response = request.execute();

                pageToken = response.getNextPageToken();
                if (pageToken == null) break;
                allPageTokens.add(pageToken); // enqueue next page token

                Thread.sleep(100);

            } catch (GoogleJsonResponseException e) {
                if (e.getStatusCode() == 403 && e.getMessage() != null && e.getMessage().contains("quota")) {
                    apiKeyManager.recordQuotaExceeded(currentKey);
                    // continue to the next key
                    continue;
                } else if (e.getMessage() != null && e.getMessage().contains("commentsDisabled")) {
                    System.out.println("[INFO] Comments are disabled for this video. No comments to fetch.");
                    return;
                } else {
                    throw e; 
                }
            }
        }

        System.out.println("[INFO] Found " + allPageTokens.size() + " pages of comments. Now fetching them in parallel...");

        Queue<String> pageTokenQueue = new ConcurrentLinkedQueue<>();
        pageTokenQueue.addAll(allPageTokens); // all tokens are non-null ("" for first page)
        
        PriorityBlockingQueue<CommentPage> completedPages = new PriorityBlockingQueue<>(
            allPageTokens.size(),
            Comparator.comparingInt(a -> a.pageNumber)
        );

        AtomicInteger processedPages = new AtomicInteger(0);
        ExecutorService executor = Executors.newFixedThreadPool(numThreads);
        AtomicBoolean hasError = new AtomicBoolean(false);
        
        for (int i = 0; i < numThreads; i++) {
            final int threadId = i;
            executor.submit(() -> {
                try {
                    YouTube threadYoutubeService = new YouTube.Builder(
                            GoogleNetHttpTransport.newTrustedTransport(), JSON_FACTORY, null)
                            .setApplicationName(APPLICATION_NAME)
                            .build();

                    while (!pageTokenQueue.isEmpty() && !hasError.get()) {
                        String token = pageTokenQueue.poll();
                        if (token == null) break;

                        String tokenValue = token.isEmpty() ? null : token;

                        String currentKey = apiKeyManager.getNextAvailableKey();

                        try {
                            YouTube.CommentThreads.List request = threadYoutubeService.commentThreads()
                                    .list("snippet")
                                    .setVideoId(videoId)
                                    .setMaxResults((long) COMMENTS_PER_PAGE)
                                    .setTextFormat("plainText")
                                    .setKey(currentKey);

                            if (tokenValue != null) {
                                request.setPageToken(tokenValue);
                            }

                            CommentThreadListResponse response = request.execute();
                            List<String> pageComments = new ArrayList<>();
                            
                            for (CommentThread thread : response.getItems()) {
                                String comment = thread.getSnippet().getTopLevelComment().getSnippet().getTextDisplay();
                                String cleaned = comment.replaceAll("https?://\\S+", "").trim().replaceAll("[\n\r]+", " ");
                                if (!cleaned.isEmpty()) {
                                    pageComments.add(cleaned);
                                }

                                if (thread.getSnippet().getTotalReplyCount() > 0) {
                                    List<String> replies = fetchAllReplies(thread, apiKeyManager, threadYoutubeService);
                                    pageComments.addAll(replies);
                                }
                            }

                            int pageNum = processedPages.incrementAndGet();
                            completedPages.add(new CommentPage(pageNum, token, pageComments));
                            
                            System.out.printf("[Thread-%d] Fetched page %d with %d comments (including replies) using key %s...%n",
                                threadId, pageNum, pageComments.size(), currentKey.substring(0, 10));

                        } catch (GoogleJsonResponseException e) {
                            if (e.getStatusCode() == 403 && e.getMessage() != null && e.getMessage().contains("quota")) {
                                apiKeyManager.recordQuotaExceeded(currentKey);
                                pageTokenQueue.offer(token);
                            } else {
                                throw e;
                            }
                        }
                    }
                } catch (Exception e) {
                    hasError.set(true);
                    System.err.println("[ERROR] Thread " + threadId + " failed: " + e.getMessage());
                    e.printStackTrace();
                }
            });
        }

        executor.shutdown();
        executor.awaitTermination(180, TimeUnit.MINUTES);

        if (hasError.get()) {
            throw new Exception("One or more threads encountered errors during comment fetching.");
        }

        List<String> allComments = new ArrayList<>();
        while (!completedPages.isEmpty()) {
            CommentPage page = completedPages.poll();
            allComments.addAll(page.comments);
        }

        System.out.println("[INFO] Writing " + allComments.size() + " comments to " + outputFile);
        try (Writer writer = new OutputStreamWriter(new FileOutputStream(outputFile), StandardCharsets.UTF_8)) {
            new GsonBuilder().setPrettyPrinting().create().toJson(allComments, writer);
        }
    }
}

