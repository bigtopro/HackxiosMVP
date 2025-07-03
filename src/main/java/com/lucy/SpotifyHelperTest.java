package com.lucy;

public class SpotifyHelperTest {
    public static void main(String[] args) {
        String testUrl = "https://open.spotify.com/track/12bZjnzjuJoyRslVC4BmxU";
        try {
            SpotifyHelper.SongInfo info = SpotifyHelper.getSongInfoFromSpotifyUrl(testUrl);
            System.out.println("Song Name: " + info.name);
            System.out.println("Artist: " + info.artist);
        } catch (Exception e) {
            e.printStackTrace();
        }
    }
} 