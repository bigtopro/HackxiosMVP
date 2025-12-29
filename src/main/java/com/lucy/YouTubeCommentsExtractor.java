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

import java.io.*;
import java.nio.charset.StandardCharsets;
import java.security.GeneralSecurityException;
import java.util.*;
import java.util.concurrent.*;
import com.google.api.client.googleapis.json.GoogleJsonResponseException;

public class YouTubeCommentsExtractor {

    private static final String APPLICATION_NAME = "YouTube-Comment-Extractor";
    private static final JsonFactory JSON_FACTORY = GsonFactory.getDefaultInstance();
    private static final int COMMENTS_PER_PAGE = 100;

    public static void main(String[] args) {
        if (args.length == 0) {
            System.out.println("Usage: java YouTubeCommentsExtractor <video_id> [output_file]");
            System.out.println("Example: java YouTubeCommentsExtractor dQw4w9WgXcQ");
            return;
        }

        String videoId = args[0];
        String outputFile = args.length > 1 ? args[1] : "comments/" + videoId + "_comments.json";

        // Create comments directory if it doesn't exist
        new File("comments").mkdirs();

        try {
            // Initialize API key manager with your YouTube API keys
            String[] apiKeys = YoutubeCommentScraper.API_KEYS;

            // Check if API keys are available
            if (apiKeys.length == 0) {
                System.err.println("No API keys available. Please set your YouTube API keys in environment variables.");
                return;
            }

            ApiKeyManager apiKeyManager = new ApiKeyManager(apiKeys);

            // Extract comments
            System.out.println("Extracting comments for video: " + videoId);
            fetchCommentsForVideo(videoId, outputFile, apiKeyManager);
            System.out.println("Comments saved to: " + outputFile);

        } catch (Exception e) {
            System.err.println("Error extracting comments: " + e.getMessage());
            e.printStackTrace();
        }
    }

    public static void fetchCommentsForVideo(String videoId, String outputFile, ApiKeyManager apiKeyManager) throws Exception {
        // Check if comments are disabled for the video
        YouTubeSearchHelper searchHelper = new YouTubeSearchHelper(apiKeyManager);
        YouTubeSearchHelper.VideoInfo videoInfo = searchHelper.getVideoInfoById(videoId);
        
        if (videoInfo.commentsDisabled) {
            System.out.println("Comments are disabled for video: " + videoId);
            // Create an empty JSON array file
            try (Writer writer = new OutputStreamWriter(new FileOutputStream(outputFile), StandardCharsets.UTF_8)) {
                new GsonBuilder().setPrettyPrinting().create().toJson(new ArrayList<>(), writer);
            }
            return;
        }

        System.out.printf("Video %s: %d comments%n", videoId, videoInfo.commentCount);

        // Determine number of threads based on comment count
        int recommendedThreads = getRecommendedThreads(videoInfo.commentCount);

        // Fetch comments in parallel
        YoutubeCommentScraper.fetchCommentsParallel(videoId, outputFile, recommendedThreads, apiKeyManager);
    }

    private static int getRecommendedThreads(long commentCount) {
        if (commentCount < 5000) return 2;
        if (commentCount < 20000) return 4;
        if (commentCount < 50000) return 8;
        if (commentCount < 100000) return 12;
        if (commentCount < 150000) return 16;
        return 20;
    }
}