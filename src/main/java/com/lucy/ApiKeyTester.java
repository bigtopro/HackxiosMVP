package com.lucy;

import com.google.api.client.googleapis.javanet.GoogleNetHttpTransport;
import com.google.api.client.json.JsonFactory;
import com.google.api.client.json.gson.GsonFactory;
import com.google.api.services.youtube.YouTube;
import com.google.api.services.youtube.model.VideoListResponse;

public class ApiKeyTester {
    private static final String APPLICATION_NAME = "YouTube-API-Test";
    private static final JsonFactory JSON_FACTORY = GsonFactory.getDefaultInstance();

    public static void main(String[] args) {
        try {
            String[] apiKeys = YoutubeCommentScraper.API_KEYS;

            if (apiKeys.length == 0) {
                System.out.println("No API keys found in environment variables.");
                System.out.println("Please set YOUTUBE_API_KEY_1, YOUTUBE_API_KEY_2, etc. in your environment or .env file");
                return;
            }

            // Test each API key
            for (String apiKey : apiKeys) {
                try {
                    System.out.println("\nTesting API key: " + apiKey);

                    YouTube youtube = new YouTube.Builder(
                            GoogleNetHttpTransport.newTrustedTransport(),
                            JSON_FACTORY,
                            null)
                            .setApplicationName(APPLICATION_NAME)
                            .build();

                    // Try to get info about a known video (using YouTube's most viewed video)
                    YouTube.Videos.List request = youtube.videos()
                            .list("snippet")
                            .setId("jNQXAC9IVRw")  // First YouTube video ever
                            .setKey(apiKey);

                    VideoListResponse response = request.execute();

                    if (response.getItems() != null && !response.getItems().isEmpty()) {
                        System.out.println("✓ API key is valid!");
                        System.out.println("Video title: " + response.getItems().get(0).getSnippet().getTitle());
                    } else {
                        System.out.println("✗ API key returned no results");
                    }

                } catch (Exception e) {
                    System.out.println("✗ API key is invalid: " + e.getMessage());
                }
            }
        } catch (Exception e) {
            e.printStackTrace();
        }
    }
}