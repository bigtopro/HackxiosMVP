package com.lucy;

import com.google.api.client.googleapis.json.GoogleJsonResponseException;
import com.google.gson.Gson;
import com.google.gson.reflect.TypeToken;

import java.io.*;
import java.util.ArrayList;
import java.util.HashMap;
import java.util.List;
import java.util.Map;

public class SpotifyToYouTubeCommentScraper {
    private static final String SONG_LIST_FILE = "songList.txt";
    private static final String[] YOUTUBE_API_KEYS = YoutubeCommentScraper.API_KEYS;
    private static final String PROGRESS_FILE = "scraping_progress.txt";

    private static class SongProgress {
        final int index;
        final String spotifyUrl;
        final String name;
        final String artist;
        boolean completed;
        String videoId;
        long commentCount;
        boolean commentsDisabled;

        SongProgress(int index, String spotifyUrl, String name, String artist) {
            this.index = index;
            this.spotifyUrl = spotifyUrl;
            this.name = name;
            this.artist = artist;
            this.completed = false;
        }
    }

    public static void main(String[] args) throws Exception {
        ApiKeyManager apiKeyManager = new ApiKeyManager(YOUTUBE_API_KEYS);
        YouTubeSearchHelper searchHelper = new YouTubeSearchHelper(apiKeyManager);

        String inputFile = SONG_LIST_FILE;
        boolean isTabSeparated = false;
        if (args.length > 0) {
            inputFile = args[0];
            if (inputFile.equals("missing_comments.txt")) {
                isTabSeparated = true;
            }
        }

        List<String> trackUrls = readSongListFlexible(inputFile, isTabSeparated);
        int totalTracks = trackUrls.size();

        Map<String, Map<String, Object>> videoCache = new HashMap<>();
        File cacheFile = new File("video_cache.json");
        if (cacheFile.exists()) {
            try (Reader reader = new FileReader(cacheFile)) {
                videoCache = new Gson().fromJson(reader, new TypeToken<Map<String, Map<String, Object>>>() {
                }.getType());
            } catch (Exception e) {
                System.err.println("[WARN] Could not read video_cache.json: " + e.getMessage());
            }
        }

        List<SongProgress> progress = loadProgress();
        if (progress.isEmpty()) {
            for (int i = 0; i < trackUrls.size(); i++) {
                String url = trackUrls.get(i);
                try {
                    SpotifyHelper.SongInfo info = SpotifyHelper.getSongInfoFromSpotifyUrl(url);
                    if (info != null) {
                        progress.add(new SongProgress(i + 1, url, info.name, info.artist));
                    }
                } catch (Exception e) {
                    System.err.printf("[ERROR] Failed to get Spotify info for URL: %s. Error: %s%n", url, e.getMessage());
                }
            }
            saveProgress(progress);
        }

        for (SongProgress song : progress) {
            if (song.completed) {
                System.out.printf("[INFO] Skipping completed song %d/%d: %s - %s%n",
                        song.index, totalTracks, song.name, song.artist);
                continue;
            }

            String cacheKey = song.name + " " + song.artist;
            if (videoCache.containsKey(cacheKey)) {
                Map<String, Object> cached = videoCache.get(cacheKey);
                if (cached != null && cached.containsKey("videoId")) {
                    System.out.printf("[INFO] Skipping song %s - already processed and in cache.%n", cacheKey);
                    song.completed = true;
                    saveProgress(progress);
                    continue;
                }
            }
            
            try {
                System.out.printf("[INFO] Processing song %d/%d: %s - %s%n",
                        song.index, totalTracks, song.name, song.artist);

                YouTubeSearchHelper.VideoInfo videoInfo = searchHelper.getVideoInfo(song.name, song.artist);
                song.videoId = videoInfo.videoId;
                song.commentCount = videoInfo.commentCount;
                song.commentsDisabled = videoInfo.commentsDisabled;

                if (song.commentsDisabled) {
                    System.out.println("[WARN] Comments are disabled for video: " + song.videoId);
                    song.completed = true;
                    saveProgress(progress);
                    continue;
                }

                System.out.printf("[INFO] Video %s: %d comments%n",
                        song.videoId, song.commentCount);

                String safeName = (song.name + "_" + song.artist).replaceAll("[^a-zA-Z0-9]", "_");
                String outputFile = "comments/" + safeName + ".json";
                new File("comments").mkdirs();

                int recommendedThreads = getRecommendedThreads(song.commentCount);

                YoutubeCommentScraper.fetchCommentsParallel(
                        song.videoId, outputFile, recommendedThreads, apiKeyManager);
                song.completed = true;
                saveProgress(progress);

            } catch (GoogleJsonResponseException e) {
                if (e.getStatusCode() == 403 && e.getMessage().contains("quota")) {
                    System.err.println("[ERROR] A quota error occurred that was not handled by the key manager. This may indicate all keys are exhausted.");
                } else {
                    System.err.println("[ERROR] Google API Error: " + e.getDetails().getMessage());
                }
                e.printStackTrace();
            } catch (Exception e) {
                System.err.println("[ERROR] Failed to process song: " + e.getMessage());
                e.printStackTrace();
            }
        }
    }

    private static int getRecommendedThreads(long commentCount) {
        if (commentCount < 5000) return 2;
        if (commentCount < 20000) return 4;
        if (commentCount < 50000) return 8;
        if (commentCount < 100000) return 12;
        if (commentCount < 150000) return 16;
        return 20;
    }

    private static List<SongProgress> loadProgress() {
        File file = new File(PROGRESS_FILE);
        if (!file.exists()) {
            return new ArrayList<>();
        }
        try (BufferedReader reader = new BufferedReader(new FileReader(file))) {
            List<SongProgress> progress = new ArrayList<>();
            String line;
            while ((line = reader.readLine()) != null) {
                String[] parts = line.split("\t");
                if (parts.length >= 4) {
                    SongProgress song = new SongProgress(
                            Integer.parseInt(parts[0]),
                            parts[1],
                            parts[2],
                            parts[3]
                    );
                    if (parts.length > 4) {
                        song.completed = Boolean.parseBoolean(parts[4]);
                        if (parts.length > 5 && parts[5] != null && !parts[5].equals("null")) song.videoId = parts[5];
                        if (parts.length > 6) song.commentCount = Long.parseLong(parts[6]);
                        if (parts.length > 7) song.commentsDisabled = Boolean.parseBoolean(parts[7]);
                    }
                    progress.add(song);
                }
            }
            return progress;
        } catch (IOException | NumberFormatException e) {
            System.err.println("[WARN] Could not parse progress file, starting from scratch. Error: " + e.getMessage());
            return new ArrayList<>();
        }
    }

    private static void saveProgress(List<SongProgress> progress) {
        try (BufferedWriter writer = new BufferedWriter(new FileWriter(PROGRESS_FILE))) {
            for (SongProgress song : progress) {
                writer.write(String.format("%d\t%s\t%s\t%s\t%b\t%s\t%d\t%b%n",
                        song.index, song.spotifyUrl, song.name, song.artist, song.completed,
                        song.videoId, song.commentCount, song.commentsDisabled));
            }
        } catch (IOException e) {
            System.err.println("[ERROR] Failed to save progress: " + e.getMessage());
        }
    }

    public static List<String> readSongListFlexible(String filename, boolean isTabSeparated) throws IOException {
        List<String> songs = new ArrayList<>();
        try (BufferedReader reader = new BufferedReader(new FileReader(filename))) {
            String line;
            while ((line = reader.readLine()) != null) {
                if (isTabSeparated) {
                    String[] parts = line.split("\t");
                    if (parts.length > 0) {
                        songs.add(parts[0]);
                    }
                } else {
                    songs.add(line);
                }
            }
        }
        return songs;
    }
} 