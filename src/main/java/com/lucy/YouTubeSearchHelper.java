package com.lucy;

import com.google.api.client.googleapis.javanet.GoogleNetHttpTransport;
import com.google.api.client.json.JsonFactory;
import com.google.api.client.json.gson.GsonFactory;
import com.google.api.services.youtube.YouTube;
import com.google.api.services.youtube.model.SearchListResponse;
import com.google.api.services.youtube.model.SearchResult;

import java.io.IOException;
import java.security.GeneralSecurityException;
import java.util.List;

public class YouTubeSearchHelper {
    private static final String APPLICATION_NAME = "YouTube-Comment-Fetcher";
    private static final JsonFactory JSON_FACTORY = GsonFactory.getDefaultInstance();

    public static String getTopVideoId(String query, String apiKey) throws GeneralSecurityException, IOException {
        YouTube youtubeService = new YouTube.Builder(
                GoogleNetHttpTransport.newTrustedTransport(), JSON_FACTORY, null)
                .setApplicationName(APPLICATION_NAME)
                .build();

        YouTube.Search.List search = youtubeService.search().list("snippet");
        search.setQ(query);
        search.setType("video");
        search.setOrder("viewCount");
        search.setMaxResults(1L);
        search.setKey(apiKey);

        SearchListResponse response = search.execute();
        List<SearchResult> results = response.getItems();
        if (results != null && !results.isEmpty()) {
            return results.get(0).getId().getVideoId();
        }
        return null;
    }
} 