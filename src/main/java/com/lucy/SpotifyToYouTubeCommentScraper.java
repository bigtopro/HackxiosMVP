package com.lucy;

import com.google.api.client.googleapis.json.GoogleJsonResponseException;
import se.michaelthelin.spotify.exceptions.SpotifyWebApiException;

import java.io.*;
import java.security.GeneralSecurityException;
import java.util.List;
import java.util.concurrent.TimeUnit;
import java.util.stream.Collectors;

public class SpotifyToYouTubeCommentScraper {
    private static final String SONG_LIST_FILE = "songList.txt";
    private static final String[] YOUTUBE_API_KEYS = YoutubeCommentScraper.API_KEYS;
    private static final int COMMENTS_PER_SONG = 226800; // Max YouTube comment limit

    public static void main(String[] args) throws Exception {
        System.out.println("[DEBUG] Starting main method...");
        System.out.println("[DEBUG] Attempting to read song list from file: " + SONG_LIST_FILE);
        List<String> trackUrls = readSongList(SONG_LIST_FILE);
        System.out.println("[DEBUG] Successfully read " + trackUrls.size() + " tracks from song list.");
        int totalTracks = trackUrls.size();
        int startIndex = 0;
        File checkpointFile = new File("resume_checkpoint.txt");
        if (checkpointFile.exists()) {
            try (BufferedReader br = new BufferedReader(new FileReader(checkpointFile))) {
                String line = br.readLine();
                if (line != null && !line.isEmpty()) {
                    startIndex = Integer.parseInt(line.trim());
                    System.out.println("[RESUME] Resuming from song index: " + startIndex);
                }
            } catch (Exception e) {
                System.err.println("[RESUME] Failed to read checkpoint file, starting from beginning.");
            }
        }
        int trackNum = startIndex + 1;
        System.out.println("Starting processing of " + totalTracks + " tracks from " + SONG_LIST_FILE);
        // Ensure the comments directory exists
        File commentsDir = new File("comments");
        if (!commentsDir.exists()) {
            boolean created = commentsDir.mkdirs();
            if (created) {
                System.out.println("[INFO] Created directory: comments");
            }
        }
        ApiKeyManager keyManager = new ApiKeyManager(YOUTUBE_API_KEYS);
        for (int i = startIndex; i < trackUrls.size(); i++) {
            String url = trackUrls.get(i);
            System.out.println("\n=== [" + trackNum + "/" + totalTracks + "] Processing Spotify URL: " + url + " ===");
            try {
                // 1. Get song info from Spotify
                System.out.println("Fetching song info from Spotify...");
                SpotifyHelper.SongInfo info = SpotifyHelper.getSongInfoFromSpotifyUrl(url);
                String query = info.name + " " + info.artist;
                System.out.println("[Spotify] Song: '" + info.name + "' | Artist: '" + info.artist + "'");
                System.out.println("Searching YouTube for: '" + query + "'");

                boolean success = false;
                while (!success) {
                    String apiKey = keyManager.getCurrentKey();
                    try {
                        String videoId = YouTubeSearchHelper.getTopVideoId(query, apiKey);
                        if (videoId == null) {
                            System.err.println("[YouTube] No video found for: '" + query + "'");
                            break;
                        }
                        System.out.println("[YouTube] Top video ID: " + videoId + " (https://www.youtube.com/watch?v=" + videoId + ")");

                        // 3. Scrape comments for this video
                        String safeName = (info.name + "_" + info.artist).replaceAll("[^a-zA-Z0-9]", "_");
                        String outputFile = "comments/" + safeName + ".txt";
                        System.out.println("[Scraper] Saving up to " + COMMENTS_PER_SONG + " comments to: '" + outputFile + "'");
                        YoutubeCommentScraper.fetchCommentsForVideo(videoId, outputFile, apiKey, COMMENTS_PER_SONG);
                        System.out.println("[Scraper] Done: " + outputFile);
                        success = true;
                    } catch (GoogleJsonResponseException e) {
                        String response = e.getContent();
                        if (response != null && response.contains("quotaExceeded")) {
                            System.err.println("[Quota] API key quota exceeded. Switching to next key.");
                            keyManager.markCurrentKeyExhausted();
                            if (keyManager.allKeysExhausted()) {
                                System.err.println("[Quota] All API keys exhausted. Saving checkpoint and waiting 1 hour before retrying...");
                                // Save checkpoint
                                try (BufferedWriter bw = new BufferedWriter(new FileWriter("resume_checkpoint.txt"))) {
                                    bw.write(Integer.toString(i));
                                } catch (IOException ioe) {
                                    System.err.println("[Checkpoint] Failed to write checkpoint file: " + ioe.getMessage());
                                }
                                System.err.println("[Checkpoint] Progress saved. You can resume tomorrow.");
                                Thread.sleep(60 * 60 * 1000); // 1 hour
                                keyManager.resetAllKeys();
                            } else {
                                keyManager.switchToNextKey();
                            }
                        } else {
                            throw e;
                        }
                    }
                    // Throttle requests to avoid burst
                    TimeUnit.SECONDS.sleep(2);
                }
            } catch (SpotifyWebApiException e) {
                System.err.println("[ERROR] Spotify API error for URL: '" + url + "' - " + e.getMessage());
                e.printStackTrace(System.err);
            } catch (GoogleJsonResponseException e) {
                System.err.println("[ERROR] YouTube API error for query: '" + url + "' - " + e.getDetails());
                e.printStackTrace(System.err);
            } catch (Exception e) {
                System.err.println("[ERROR] General error for URL: '" + url + "' - " + e.getMessage());
                e.printStackTrace(System.err);
            }
            trackNum++;
        }
        // Remove checkpoint file if finished
        if (checkpointFile.exists()) checkpointFile.delete();
        System.out.println("\nAll tracks processed. Check output files for results.");
    }

    public static List<String> readSongList(String filename) throws IOException {
        try (BufferedReader br = new BufferedReader(new FileReader(filename))) {
            return br.lines().map(String::trim).filter(line -> !line.isEmpty()).collect(Collectors.toList());
        }
    }
} 