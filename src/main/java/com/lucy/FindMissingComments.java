package com.lucy;

import java.io.File;
import java.io.IOException;
import java.util.List;

public class FindMissingComments {
    public static void main(String[] args) throws IOException {
        String songListFile = "songList.txt";
        if (args.length > 0) {
            songListFile = args[0];
        }

        List<String> songs = SpotifyToYouTubeCommentScraper.readSongListFlexible(songListFile, songListFile.equals("missing_comments.txt"));
        System.out.println("Found " + songs.size() + " songs in list");

        int missing = 0;
        for (String url : songs) {
            String[] parts = url.split("\t");
            String songName = parts.length > 1 ? parts[0] : url;
            String safeName = songName.replaceAll("[^a-zA-Z0-9]", "_");
            File commentFile = new File("comments/" + safeName + ".json");
            
            if (!commentFile.exists()) {
                System.out.println("Missing comments for: " + url);
                missing++;
            }
        }

        System.out.println("\nTotal missing: " + missing + " out of " + songs.size());
    }
} 