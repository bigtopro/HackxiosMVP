package com.lucy;

import java.io.IOException;
import java.util.List;

public class DebugTest {
    public static void main(String[] args) throws Exception {
        String songListFile = "songList.txt";
        if (args.length > 0) {
            songListFile = args[0];
        }

        List<String> songs = SpotifyToYouTubeCommentScraper.readSongListFlexible(songListFile, songListFile.equals("missing_comments.txt"));
        System.out.println("Found " + songs.size() + " songs in list");

        ApiKeyManager apiKeyManager = new ApiKeyManager(YoutubeCommentScraper.API_KEYS);
        YouTubeSearchHelper searchHelper = new YouTubeSearchHelper(apiKeyManager);

        for (String url : songs) {
            try {
                SpotifyHelper.SongInfo info = SpotifyHelper.getSongInfoFromSpotifyUrl(url);
                System.out.println("\nProcessing: " + info.name + " by " + info.artist);
                
                YouTubeSearchHelper.VideoInfo videoInfo = searchHelper.getVideoInfo(info.name, info.artist);
                if (videoInfo.commentsDisabled) {
                    System.out.println("Comments are disabled for video: " + videoInfo.videoId);
                } else {
                    System.out.printf("Found video %s with %d comments%n", 
                        videoInfo.videoId, videoInfo.commentCount);
                }
                
                Thread.sleep(1000); // Avoid rate limits
            } catch (Exception e) {
                System.err.println("Error processing " + url + ": " + e.getMessage());
            }
        }
    }
} 