import praw
import prawcore
import datetime
import os
from dotenv import load_dotenv
from openai import OpenAI
import json

# Load environment variables from .env file if it exists
if os.path.exists(".env"):
    load_dotenv()

# Function to get environment variables
def get_env_variable(var_name):
    return os.getenv(var_name)

# Set up Reddit API client
try:
    reddit = praw.Reddit(
        client_id=os.environ.get("REDDIT_CLIENT_ID"),
        client_secret=os.environ.get("REDDIT_CLIENT_SECRET"),
        user_agent=os.environ.get("REDDIT_USER_AGENT")
    )
    # Verify the credentials by making a simple API call
    reddit.user.me()
except prawcore.exceptions.ResponseException as e:
    print(f"Error authenticating with Reddit API: {str(e)}")
    print("Please check your Reddit API credentials in the .env file.")
    exit(1)
except prawcore.exceptions.OAuthException as e:
    print(f"OAuth Error: {str(e)}")
    print("Please verify your Reddit API credentials and ensure they have the correct permissions.")
    exit(1)

# Set up OpenAI client
client = OpenAI(api_key=get_env_variable("OPENAI_API_KEY"))

def search_reddit_posts(query, limit, time_filter, sort, subreddit=None):
    posts = []
    limit = min(limit, 10)  # Ensure limit is at most 10

    if subreddit:
        search_subreddit = reddit.subreddit(subreddit)
    else:
        search_subreddit = reddit.subreddit("all")

    if sort == "relevance":
        submissions = search_subreddit.search(query, sort=sort, time_filter=time_filter, limit=10)
    elif sort == "hot":
        submissions = search_subreddit.search(query, sort="hot", limit=10)
    elif sort == "new":
        submissions = search_subreddit.search(query, sort="new", limit=10)
    elif sort == "top":
        submissions = search_subreddit.search(query, sort="top", time_filter=time_filter, limit=10)
    else:
        raise ValueError("Invalid sort parameter")

    for submission in submissions:
        created_date = datetime.datetime.fromtimestamp(submission.created_utc).strftime('%Y-%m-%d %H:%M:%S')
        posts.append({
            "title": submission.title,
            "url": f"https://www.reddit.com{submission.permalink}",
            "score": submission.score,
            "num_comments": submission.num_comments,
            "created_date": created_date,
            "subreddit": submission.subreddit.display_name
        })

    # Sort posts by score in descending order and limit to the requested number
    posts.sort(key=lambda x: x['score'], reverse=True)
    return posts[:limit]

def analyze_reddit_posts(user_prompt):
    functions = [
        {
            "name": "search_reddit_posts",
            "description": "Search for max 10 posts across all subreddits or a specific subreddit, ordered by score (upvotes)",
            "parameters": {
                "type": "object",
                "properties": {
                    "query": {
                        "type": "string",
                        "description": "Search query to find relevant posts"
                    },
                    "limit": {
                        "type": "integer",
                        "description": "Maximum number of posts to fetch (max 10)"
                    },
                    "time_filter": {
                        "type": "string",
                        "enum": ["hour", "day", "week", "month", "year", "all"],
                        "description": "Time filter for search results"
                    },
                    "sort": {
                        "type": "string",
                        "enum": ["relevance", "hot", "new", "top"],
                        "description": "Initial sorting method for search results before ordering by score"
                    },
                    "subreddit": {
                        "type": "string",
                        "description": "Specific subreddit to search in (optional)"
                    }
                },
                "required": ["query", "limit", "time_filter", "sort"]
            }
        }
    ]

    messages = [
        {"role": "system", "content": "You are a helpful assistant that analyzes Reddit posts. Use the search_reddit_posts function to search for relevant posts across all subreddits or a specific subreddit based on the user's request. The posts will be ordered by score (upvotes). Always include the URL, subreddit name, created date and score for each Reddit post in your analysis. If the user specifies a subreddit, make sure to include it in the function call."},
        {"role": "user", "content": user_prompt}
    ]

    response = client.chat.completions.create(
        model=os.environ.get("MODEL"),
        messages=messages,
        functions=functions,
        function_call="auto"
    )

    response_message = response.choices[0].message

    if response_message.function_call:
        function_name = response_message.function_call.name
        function_args = json.loads(response_message.function_call.arguments)
        
        if function_name == "search_reddit_posts":
            posts = search_reddit_posts(**function_args)
            
            messages.append(response_message)
            messages.append(
                {
                    "role": "function",
                    "name": function_name,
                    "content": json.dumps(posts)
                }
            )
            
            second_response = client.chat.completions.create(
                model=os.environ.get("MODEL"),
                messages=messages
            )
            
            return second_response.choices[0].message.content
    
    return response_message.content

def main():
    print("Reddit Post Analyzer")
    print("Ask me to analyze Reddit posts, and I'll search and analyze them for you!")
    print("You can specify a subreddit if you want to focus on a particular community.")

    user_prompt = input("What would you like to know about Reddit posts? ")
    
    print("Analyzing Reddit posts...")
    analysis = analyze_reddit_posts(user_prompt)
    
    print("\nAnalysis Results:")
    print(analysis)

if __name__ == "__main__":
    main()
