package com.lucy;

import se.michaelthelin.spotify.SpotifyApi;
import se.michaelthelin.spotify.SpotifyHttpManager;
import se.michaelthelin.spotify.exceptions.SpotifyWebApiException;
import se.michaelthelin.spotify.model_objects.specification.Track;
import se.michaelthelin.spotify.requests.authorization.client_credentials.ClientCredentialsRequest;
import se.michaelthelin.spotify.requests.data.tracks.GetTrackRequest;

import java.io.IOException;
import java.net.URI;
import java.util.regex.Matcher;
import java.util.regex.Pattern;

public class SpotifyHelper {
    private static final String clientId = "72c7ae1e70f141679c46905b3bc8f361";
    private static final String clientSecret = "e2e8e0f6b7e7474c877010be1d251f52";
    private static final SpotifyApi spotifyApi = new SpotifyApi.Builder()
            .setClientId(clientId)
            .setClientSecret(clientSecret)
            .build();

    public static class SongInfo {
        public final String name;
        public final String artist;
        public SongInfo(String name, String artist) {
            this.name = name;
            this.artist = artist;
        }
    }

    public static SongInfo getSongInfoFromSpotifyUrl(String url) throws IOException, SpotifyWebApiException, org.apache.hc.core5.http.ParseException {
        String trackId = extractTrackId(url);
        if (trackId == null) throw new IllegalArgumentException("Invalid Spotify track URL: " + url);

        // Authenticate
        ClientCredentialsRequest clientCredentialsRequest = spotifyApi.clientCredentials().build();
        try {
            se.michaelthelin.spotify.model_objects.credentials.ClientCredentials clientCredentials = clientCredentialsRequest.execute();
            spotifyApi.setAccessToken(clientCredentials.getAccessToken());
        } catch (Exception e) {
            throw new IOException("Failed to authenticate with Spotify API", e);
        }

        // Fetch track info
        try {
            GetTrackRequest getTrackRequest = spotifyApi.getTrack(trackId).build();
            Track track = getTrackRequest.execute();
            String name = track.getName();
            String artist = track.getArtists()[0].getName();
            return new SongInfo(name, artist);
        } catch (IOException | SpotifyWebApiException e) {
            throw new IOException("Failed to fetch track info from Spotify API", e);
        }
    }

    private static String extractTrackId(String url) {
        // Example: https://open.spotify.com/track/12bZjnzjuJoyRslVC4BmxU
        Pattern pattern = Pattern.compile("track/([a-zA-Z0-9]+)");
        Matcher matcher = pattern.matcher(url);
        if (matcher.find()) {
            return matcher.group(1);
        }
        return null;
    }
} 