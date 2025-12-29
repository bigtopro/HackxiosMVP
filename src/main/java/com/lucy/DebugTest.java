package com.lucy;

import java.io.IOException;
import java.util.List;
import java.nio.file.Files;
import java.nio.file.Paths;

public class DebugTest {
    public static void main(String[] args) throws Exception {
        String videoListFile = "videoList.txt";
        if (args.length > 0) {
            videoListFile = args[0];
        }

        List<String> videoIds = readVideoList(videoListFile);
        System.out.println("Found " + videoIds.size() + " video IDs in list");

        String[] apiKeys = YoutubeCommentScraper.API_KEYS;
        if (apiKeys.length == 0) {
            System.out.println("No API keys available. Please set your YouTube API keys in environment variables.");
            return;
        }

        ApiKeyManager apiKeyManager = new ApiKeyManager(apiKeys);
        YouTubeSearchHelper searchHelper = new YouTubeSearchHelper(apiKeyManager);

        for (String videoId : videoIds) {
            try {
                System.out.println("\nProcessing video: " + videoId);

                YouTubeSearchHelper.VideoInfo videoInfo = searchHelper.getVideoInfoById(videoId);
                if (videoInfo.commentsDisabled) {
                    System.out.println("Comments are disabled for video: " + videoInfo.videoId);
                } else {
                    System.out.printf("Video %s has %d comments%n",
                        videoInfo.videoId, videoInfo.commentCount);
                }

                Thread.sleep(1000); // Avoid rate limits
            } catch (Exception e) {
                System.err.println("Error processing " + videoId + ": " + e.getMessage());
            }
        }
    }

    private static List<String> readVideoList(String filename) throws IOException {
        if (Files.exists(Paths.get(filename))) {
            return Files.readAllLines(Paths.get(filename));
        } else {
            // Return a default list or empty list
            System.out.println("Video list file not found: " + filename);
            return new java.util.ArrayList<>();
        }
    }
}