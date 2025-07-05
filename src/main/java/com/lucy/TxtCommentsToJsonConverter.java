package com.lucy;

import com.google.gson.Gson;
import com.google.gson.GsonBuilder;
import java.io.*;
import java.nio.charset.StandardCharsets;
import java.util.ArrayList;
import java.util.List;
import java.util.regex.Matcher;
import java.util.regex.Pattern;

public class TxtCommentsToJsonConverter {
    public static class CommentObj {
        public int index;
        public String text;
        public CommentObj(int index, String text) {
            this.index = index;
            this.text = text;
        }
    }

    public static void main(String[] args) throws IOException {
        File commentsDir = new File("comments");
        if (!commentsDir.exists() || !commentsDir.isDirectory()) {
            System.err.println("[ERROR] 'comments' directory not found.");
            return;
        }
        File[] txtFiles = commentsDir.listFiles((dir, name) -> name.endsWith(".txt"));
        if (txtFiles == null || txtFiles.length == 0) {
            System.err.println("[INFO] No .txt files found in 'comments' directory.");
            return;
        }
        Gson gson = new GsonBuilder().setPrettyPrinting().create();
        Pattern bracePattern = Pattern.compile("^\\{(.*)\\}$");
        for (File txtFile : txtFiles) {
            List<String> comments = new ArrayList<>();
            try (BufferedReader br = new BufferedReader(new InputStreamReader(new FileInputStream(txtFile), StandardCharsets.UTF_8))) {
                String line;
                while ((line = br.readLine()) != null) {
                    line = line.trim();
                    if (line.isEmpty()) continue;
                    Matcher m = bracePattern.matcher(line);
                    String commentText = m.matches() ? m.group(1).trim() : line;
                    comments.add(commentText);
                }
            }
            String jsonFileName = txtFile.getName().replaceFirst("\\.txt$", ".json");
            File jsonFile = new File(commentsDir, jsonFileName);
            try (Writer writer = new OutputStreamWriter(new FileOutputStream(jsonFile), StandardCharsets.UTF_8)) {
                gson.toJson(comments, writer);
            }
            System.out.println("Converted: " + txtFile.getName() + " -> " + jsonFileName);
        }
        System.out.println("All .txt files converted to .json in 'comments' directory.");
    }
} 