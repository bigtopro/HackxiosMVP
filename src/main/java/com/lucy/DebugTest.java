package com.lucy;

import se.michaelthelin.spotify.exceptions.SpotifyWebApiException;
import java.io.IOException;
import java.util.List;

public class DebugTest {
    public static void main(String[] args) throws Exception {
        // 1. Test reading songList.txt
        System.out.println("[TEST] Reading songList.txt...");
        List<String> trackUrls = SpotifyToYouTubeCommentScraper.readSongList("songList.txt");
        System.out.println("[TEST] Found " + trackUrls.size() + " tracks.");
        for (int i = 0; i < Math.min(3, trackUrls.size()); i++) {
            System.out.println("[TEST] Track URL " + (i+1) + ": " + trackUrls.get(i));
        }

        // 2. Test SpotifyHelper
        if (!trackUrls.isEmpty()) {
            String testUrl = trackUrls.get(0);
            System.out.println("\n[TEST] Fetching song info from Spotify for: " + testUrl);
            try {
                SpotifyHelper.SongInfo info = SpotifyHelper.getSongInfoFromSpotifyUrl(testUrl);
                System.out.println("[TEST] Song: " + info.name);
                System.out.println("[TEST] Artist: " + info.artist);

                // 3. Test YouTubeSearchHelper
                String query = info.name + " " + info.artist;
                String apiKey = YoutubeCommentScraper.API_KEYS[0];
                System.out.println("\n[TEST] Searching YouTube for: '" + query + "'");
                String videoId = YouTubeSearchHelper.getTopVideoId(query, apiKey);
                if (videoId != null) {
                    System.out.println("[TEST] Top YouTube video ID: " + videoId);
                    System.out.println("[TEST] YouTube link: https://www.youtube.com/watch?v=" + videoId);

                    // 4. Test YoutubeCommentScraper.fetchCommentsForVideo
                    String outputFile = "debug_comments.txt";
                    System.out.println("\n[TEST] Fetching comments for video and saving to: " + outputFile);
                    YoutubeCommentScraper.fetchCommentsForVideo(videoId, outputFile, apiKey, 10);
                    System.out.println("[TEST] Comments fetched and saved to " + outputFile);
                } else {
                    System.out.println("[TEST] No YouTube video found for: '" + query + "'");
                }
            } catch (SpotifyWebApiException e) {
                System.err.println("[TEST][ERROR] Spotify API error: " + e.getMessage());
                e.printStackTrace(System.err);
            } catch (IOException e) {
                System.err.println("[TEST][ERROR] IO error: " + e.getMessage());
                e.printStackTrace(System.err);
            } catch (Exception e) {
                System.err.println("[TEST][ERROR] General error: " + e.getMessage());
                e.printStackTrace(System.err);
            }
        }
    }
} 