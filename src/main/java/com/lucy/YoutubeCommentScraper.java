package com.lucy;

import com.google.api.client.googleapis.javanet.GoogleNetHttpTransport;
import com.google.api.client.http.javanet.NetHttpTransport;
import com.google.api.client.json.JsonFactory;
import com.google.api.client.json.gson.GsonFactory;
import com.google.api.services.youtube.YouTube;
import com.google.api.services.youtube.model.CommentThread;
import com.google.api.services.youtube.model.CommentThreadListResponse;

import java.io.*;
import java.nio.charset.StandardCharsets;
import java.security.GeneralSecurityException;
import java.util.*;
import java.util.concurrent.*;

public class YoutubeCommentScraper {

    private static final String APPLICATION_NAME = "YouTube-Comment-Fetcher";
    private static final JsonFactory JSON_FACTORY = GsonFactory.getDefaultInstance();
    private static final String VIDEO_ID = "7wtfhZwyrcc"; // Believer - Imagine Dragons
    private static final int COMMENTS_PER_PAGE = 100;
    private static final int TOTAL_COMMENTS = 100000;
    private static final int THREAD_COUNT = 10;
    private static final String OUTPUT_FILE = "comments.txt";
    private static final String COUNT_FILE = "n.oofcomments.txt";

    private static final String[] API_KEYS = new String[]{
            "AIzaSyAARL2E6yrqKgASel63WVaTzg9Rf8gcYj0",
            "AIzaSyArsvaV7liupXFqf9GyMoL9c-nILgdetIo",
            "AIzaSyA86byuW-2u7t6vLdL2cWbKSqqJ4r2XXLQ",
            "AIzaSyCPNhuO5dIGxXZexNKJVbyDS81oBvghfBM",
            "AIzaSyBegsB2yz7T92qlp6HMfsNx2QupyR3H6rY",
            "AIzaSyCu4hyDDFk2mvHVT7Z7M4Qqi654hRv387I",
            "AIzaSyDLmy7AlEj8_T1_E8IwyYUybAV3Q-IHUmg",
            "AIzaSyDTPJs0yXc36jucXExMlLZ2yKh_i_Ncfxs",
            "AIzaSyAXojLaxjp7DHTfY8--9MHlzZW8PDdvLmw",
            "AIzaSyDHLKXgHyIiGb32YKqKyjfzQKLrtShJnKA"
    };


    private static final List<String> preloadPageTokens = Collections.synchronizedList(new ArrayList<>());

    private static int skipPages = 0;
    private static int totalSaved = 0;

    public static void main(String[] args) throws Exception {
        int existingCount = readSavedCount();
        skipPages = existingCount / COMMENTS_PER_PAGE;
        preloadPageTokens();
        runMultiKeyScraping();
        writeSavedCount(existingCount + totalSaved);
    }

    private static int readSavedCount() {
        File file = new File(COUNT_FILE);
        if (!file.exists()) return 0;
        try (BufferedReader reader = new BufferedReader(new FileReader(file))) {
            String line = reader.readLine();
            return line != null ? Integer.parseInt(line.replaceAll("[^0-9]", "")) : 0;
        } catch (IOException e) {
            e.printStackTrace();
            return 0;
        }
    }

    private static void writeSavedCount(int count) {
        try (BufferedWriter writer = new BufferedWriter(new FileWriter(COUNT_FILE))) {
            writer.write("{" + count + "}");
        } catch (IOException e) {
            e.printStackTrace();
        }
    }

    private static void preloadPageTokens() throws GeneralSecurityException, IOException {
        YouTube youtubeService = new YouTube.Builder(
                GoogleNetHttpTransport.newTrustedTransport(), JSON_FACTORY, null)
                .setApplicationName(APPLICATION_NAME)
                .build();

        String pageToken = null;
        int pagesToSkip = skipPages;
        for (int i = 0; i < skipPages + (TOTAL_COMMENTS / COMMENTS_PER_PAGE); i++) {
            if (pagesToSkip > 0) {
                pagesToSkip--;
            } else {
                preloadPageTokens.add(pageToken);
            }

            YouTube.CommentThreads.List request = youtubeService.commentThreads()
                    .list("snippet")
                    .setVideoId(VIDEO_ID)
                    .setMaxResults((long) COMMENTS_PER_PAGE)
                    .setTextFormat("plainText")
                    .setKey(API_KEYS[0]);

            if (pageToken != null) request.setPageToken(pageToken);

            CommentThreadListResponse response = request.execute();
            pageToken = response.getNextPageToken();
            if (pageToken == null) break;
        }
    }

    private static void runMultiKeyScraping() throws InterruptedException {
        ExecutorService executor = Executors.newFixedThreadPool(THREAD_COUNT);

        for (int i = 0; i < THREAD_COUNT; i++) {
            int threadIndex = i;
            executor.submit(() -> {
                try {
                    YouTube youtubeService = new YouTube.Builder(
                            GoogleNetHttpTransport.newTrustedTransport(), JSON_FACTORY, null)
                            .setApplicationName(APPLICATION_NAME)
                            .build();

                    int pagesPerThread = (TOTAL_COMMENTS / COMMENTS_PER_PAGE) / THREAD_COUNT;
                    for (int j = 0; j < pagesPerThread; j++) {
                        int tokenIndex = threadIndex + j * THREAD_COUNT;
                        if (tokenIndex >= preloadPageTokens.size()) break;

                        String pageToken = preloadPageTokens.get(tokenIndex);
                        YouTube.CommentThreads.List request = youtubeService.commentThreads()
                                .list("snippet")
                                .setVideoId(VIDEO_ID)
                                .setMaxResults((long) COMMENTS_PER_PAGE)
                                .setTextFormat("plainText")
                                .setKey(API_KEYS[threadIndex]);

                        if (pageToken != null) request.setPageToken(pageToken);

                        CommentThreadListResponse response = request.execute();
                        for (CommentThread thread : response.getItems()) {
                            String comment = thread.getSnippet().getTopLevelComment().getSnippet().getTextDisplay();
                            String formatted = formatComment(comment);
                            if (!formatted.isEmpty()) {
                                appendCommentToFile(formatted);
                                incrementTotalSaved();
                            }
                        }
                    }
                } catch (Exception e) {
                    System.err.println("Thread " + threadIndex + " error: " + e.getMessage());
                }
            });
        }

        executor.shutdown();
        executor.awaitTermination(30, TimeUnit.MINUTES);
    }

    private static synchronized void incrementTotalSaved() {
        totalSaved++;
    }

    private static String formatComment(String comment) {
        String noUrls = comment.replaceAll("https?://\\S+", "");
        return "{" + noUrls.trim().replaceAll("[\n\r]+", " ") + "}";
    }

    private static synchronized void appendCommentToFile(String comment) {
        try (OutputStreamWriter osw = new OutputStreamWriter(new FileOutputStream(OUTPUT_FILE, true), StandardCharsets.UTF_8);
             BufferedWriter bw = new BufferedWriter(osw);
             PrintWriter out = new PrintWriter(bw)) {
            out.println(comment);
        } catch (IOException e) {
            e.printStackTrace();
        }
    }
}
