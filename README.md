# YouTube Comments Extractor

A Java-based system for extracting comments directly from YouTube videos using the YouTube Data API. Given a YouTube video ID, the system efficiently extracts all comments (including replies) and saves them to a JSON file.

## Features

- **Direct YouTube video ID input**: Provide YouTube video IDs directly
- **Multi-threaded comment extraction**: Fetches comments in parallel for efficiency
- **API key management**: Rotates through multiple API keys with rate limiting
- **Comment cleaning**: Removes URLs and normalizes comment text
- **Reply inclusion**: Fetches both top-level comments and their replies
- **Caching**: Caches video information to minimize API calls

## Prerequisites

- Java 8 or higher
- Maven
- YouTube Data API v3 keys

## Setup

1. Clone the repository
2. Add your YouTube API keys to the `.env` file
3. Add your YouTube video IDs to `videoList.txt` (one per line)

## Configuration

### API Keys
Add your YouTube Data API v3 keys to the `.env` file:
```
YOUTUBE_API_KEY_1=your_actual_api_key_here
YOUTUBE_API_KEY_2=your_actual_api_key_here
YOUTUBE_API_KEY_3=your_actual_api_key_here
YOUTUBE_API_KEY_4=your_actual_api_key_here
# Add more keys as needed (up to YOUTUBE_API_KEY_20)
```

The system uses multiple API keys to handle rate limiting and quotas effectively. The more keys you provide, the more efficient the extraction process will be.

## Usage

### Extract Comments for a Single Video
```bash
mvn compile
mvn exec:java -Dexec.mainClass="com.lucy.YouTubeCommentsExtractor" -Dexec.args="VIDEO_ID [OUTPUT_FILE]"
```

Example:
```bash
mvn exec:java -Dexec.mainClass="com.lucy.YouTubeCommentsExtractor" -Dexec.args="dQw4w9WgXcQ"
```

### Extract Comments for Multiple Videos
Add video IDs to `videoList.txt` (one per line), then run:
```bash
mvn exec:java -Dexec.mainClass="com.lucy.YouTubeCommentsExtractor" -Dexec.args="videoList.txt"
```

### Other Utilities

- `ApiKeyTester.java` - Test your YouTube API keys
- `SampleCommentsAggregator.java` - Aggregate sampled comments from all files
- `FindMissingComments.java` - Find videos without comment files
- `TxtCommentsToJsonConverter.java` - Convert text files to JSON format
- `DebugTest.java` - Debug the extraction pipeline

## Output

Comments are saved in the `comments/` directory as JSON files with the naming convention `{VIDEO_ID}_comments.json`. Each file contains an array of comment strings.

## Project Structure

```
src/
├── main/java/com/lucy/
│   ├── ApiKeyManager.java          # Manages YouTube API keys with rate limiting
│   ├── ApiKeyTester.java           # Tests YouTube API keys
│   ├── DebugTest.java              # Debugging utility
│   ├── FindMissingComments.java    # Finds missing comment files
│   ├── SampleCommentsAggregator.java # Aggregates sampled comments
│   ├── TxtCommentsToJsonConverter.java # Converts text to JSON
│   ├── VideoCache.java             # Caches video search results
│   ├── YouTubeCommentsExtractor.java # Main extractor for YouTube videos
│   ├── YouTubeSearchHelper.java    # YouTube video information helper
│   └── YoutubeCommentScraper.java  # Core comment extraction
```

## Files

- `videoList.txt` - Input file with YouTube video IDs (one per line)
- `comments/` - Output directory for extracted comments
- `.env` - Environment file for API keys (not committed to git)