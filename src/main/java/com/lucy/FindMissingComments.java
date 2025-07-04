package com.lucy;

import se.michaelthelin.spotify.exceptions.SpotifyWebApiException;
import java.io.*;
import java.util.*;

public class FindMissingComments {
    public static void main(String[] args) throws Exception {
        String songListFile = "songList.txt";
        String commentsDir = "comments";
        List<String> trackUrls = SpotifyToYouTubeCommentScraper.readSongList(songListFile);
        Set<String> existingFiles = new HashSet<>();
        File dir = new File(commentsDir);
        if (dir.exists() && dir.isDirectory()) {
            for (File f : Objects.requireNonNull(dir.listFiles())) {
                if (f.isFile()) existingFiles.add(f.getName());
            }
        } else {
            System.err.println("[ERROR] Comments directory does not exist: " + commentsDir);
            return;
        }

        List<String> missing = new ArrayList<>();
        int idx = 0;
        int total = trackUrls.size();
        for (String url : trackUrls) {
            idx++;
            if (idx % 10 == 1 || idx == total) {
                System.out.println("[Progress] Processing song " + idx + " of " + total);
            }
            try {
                SpotifyHelper.SongInfo info = SpotifyHelper.getSongInfoFromSpotifyUrl(url);
                String safeName = (info.name + "_" + info.artist).replaceAll("[^a-zA-Z0-9]", "_") + ".txt";
                if (!existingFiles.contains(safeName)) {
                    missing.add("[" + idx + "] " + info.name + " - " + info.artist + " (" + url + ")");
                }
            } catch (SpotifyWebApiException e) {
                System.err.println("[ERROR] Spotify API error for URL: '" + url + "' - " + e.getMessage());
            } catch (Exception e) {
                System.err.println("[ERROR] General error for URL: '" + url + "' - " + e.getMessage());
            }
        }

        System.out.println("\n=== Songs missing comments files ===");
        // Write missing songs and their Spotify links to a new file
        String missingFile = "missing_comments.txt";
        try (BufferedWriter bw = new BufferedWriter(new FileWriter(missingFile))) {
            for (String s : missing) {
                System.out.println(s);
                // Extract the song name, artist, and Spotify URL from the string
                int startIdx = s.lastIndexOf("(");
                int endIdx = s.lastIndexOf(")");
                String url = (startIdx != -1 && endIdx != -1 && endIdx > startIdx) ? s.substring(startIdx + 1, endIdx) : "";
                String details = s.substring(0, startIdx != -1 ? startIdx : s.length()).trim();
                // details format: [idx] name - artist
                int bracketIdx = details.indexOf("]");
                String nameArtist = (bracketIdx != -1) ? details.substring(bracketIdx + 1).trim() : details;
                String[] parts = nameArtist.split(" - ", 2);
                String name = parts.length > 0 ? parts[0].trim() : "";
                String artist = parts.length > 1 ? parts[1].trim() : "";
                // Write as: name \t artist \t url
                bw.write(name + "\t" + artist + "\t" + url);
                bw.newLine();
            }
        } catch (IOException e) {
            System.err.println("[ERROR] Failed to write missing comments file: " + e.getMessage());
        }
        System.out.println("Total missing: " + missing.size());
    }
} 