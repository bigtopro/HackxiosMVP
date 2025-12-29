package com.lucy;

import java.io.File;
import java.io.IOException;
import java.util.List;
import java.nio.file.Files;
import java.nio.file.Paths;

public class FindMissingComments {
    public static void main(String[] args) throws IOException {
        String videoListFile = "videoList.txt";
        if (args.length > 0) {
            videoListFile = args[0];
        }

        List<String> videoIds = readVideoList(videoListFile);
        System.out.println("Found " + videoIds.size() + " video IDs in list");

        int missing = 0;
        for (String videoId : videoIds) {
            File commentFile = new File("comments/" + videoId + "_comments.json");

            if (!commentFile.exists()) {
                System.out.println("Missing comments for video: " + videoId);
                missing++;
            }
        }

        System.out.println("\nTotal missing: " + missing + " out of " + videoIds.size());
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