package com.lucy;

import com.google.api.client.googleapis.javanet.GoogleNetHttpTransport;
import com.google.api.client.json.JsonFactory;
import com.google.api.client.json.gson.GsonFactory;
import com.google.api.services.youtube.YouTube;
import com.google.api.services.youtube.model.*;
import com.google.api.client.googleapis.json.GoogleJsonResponseException;

import java.io.IOException;
import java.security.GeneralSecurityException;
import java.util.Comparator;
import java.util.List;

public class YouTubeSearchHelper {
    private static final String APPLICATION_NAME = "YouTube-Comment-Scraper";
    private static final JsonFactory JSON_FACTORY = GsonFactory.getDefaultInstance();
    private final ApiKeyManager apiKeyManager;
    private VideoCache cache;

    public YouTubeSearchHelper(ApiKeyManager apiKeyManager) {
        this.apiKeyManager = apiKeyManager;
        this.cache = new VideoCache();
    }

    public static class VideoInfo {
        public final String videoId;
        public final long commentCount;
        public final boolean commentsDisabled;

        public VideoInfo(String videoId, long commentCount, boolean commentsDisabled) {
            this.videoId = videoId;
            this.commentCount = commentCount;
            this.commentsDisabled = commentsDisabled;
        }
    }

    public VideoInfo getVideoInfo(String songName, String artist) throws Exception {
        String searchQuery = String.format("%s %s", songName, artist);

        if (cache.hasValidEntry(searchQuery)) {
            VideoCache.CacheEntry entry = cache.get(searchQuery);
            return new VideoInfo(entry.videoId, entry.commentCount, entry.commentsDisabled);
        }

        String videoId = searchForVideo(searchQuery);
        if (videoId == null) {
            throw new RuntimeException("Failed to find video after retries for query: " + searchQuery);
        }

        VideoInfo info = getVideoStatistics(videoId);

        cache.put(searchQuery, new VideoCache.CacheEntry(info.videoId, info.commentCount, info.commentsDisabled));

        return info;
    }

    public VideoInfo getVideoInfoById(String videoId) throws Exception {
        return getVideoStatistics(videoId);
    }
    
    private String searchForVideo(String query) throws GeneralSecurityException, IOException, InterruptedException {
        while (true) {
            String apiKey = apiKeyManager.getNextAvailableKey();
            try {
                YouTube youtubeService = new YouTube.Builder(
                        GoogleNetHttpTransport.newTrustedTransport(), JSON_FACTORY, null)
                        .setApplicationName(APPLICATION_NAME)
                        .build();

                YouTube.Search.List search = youtubeService.search().list("snippet");
                search.setQ(query);
                search.setType("video");
                search.setOrder("relevance");
                search.setMaxResults(10L);
                search.setKey(apiKey);

                SearchListResponse response = search.execute();
                List<SearchResult> results = response.getItems();
                if (results == null || results.isEmpty()) {
                    return null;
                }
                // More sophisticated matching can be done here if needed
                return results.get(0).getId().getVideoId();

            } catch (GoogleJsonResponseException e) {
                if (e.getStatusCode() == 403 && e.getMessage() != null && e.getMessage().contains("quota")) {
                    apiKeyManager.recordQuotaExceeded(apiKey);
                    // Loop will continue and get the next available key
                } else {
                    throw e; // Re-throw other errors
                }
            }
        }
    }
    
    private VideoInfo getVideoStatistics(String videoId) throws Exception {
        while (true) {
            String apiKey = apiKeyManager.getNextAvailableKey();
            try {
                YouTube youtubeService = new YouTube.Builder(
                        GoogleNetHttpTransport.newTrustedTransport(), JSON_FACTORY, null)
                        .setApplicationName(APPLICATION_NAME)
                        .build();

                YouTube.Videos.List request = youtubeService.videos()
                        .list("statistics,snippet")
                        .setId(videoId)
                        .setKey(apiKey);

                VideoListResponse response = request.execute();
                if (response.getItems() != null && !response.getItems().isEmpty()) {
                    Video video = response.getItems().get(0);
                    if (video.getStatistics() != null && video.getStatistics().getCommentCount() != null) {
                        return new VideoInfo(videoId, video.getStatistics().getCommentCount().longValue(), false);
                    }
                }
                // If comment count is null or comments are disabled (older method)
                return new VideoInfo(videoId, 0, true);
                
            } catch (GoogleJsonResponseException e) {
                if (e.getStatusCode() == 403) {
                     if (e.getMessage() != null && e.getMessage().contains("commentsDisabled")) {
                        return new VideoInfo(videoId, 0, true);
                    } else if (e.getMessage() != null && e.getMessage().contains("quota")) {
                        apiKeyManager.recordQuotaExceeded(apiKey);
                        // Loop will get next key
                    } else {
                        throw e;
                    }
                } else {
                     throw e;
                }
            }
        }
    }

    private String normalize(String s) {
        return s.toLowerCase().replaceAll("[^a-z0-9\\s]", "").trim();
    }
} 