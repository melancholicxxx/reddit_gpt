import praw
import prawcore
import datetime
import os
from dotenv import load_dotenv
from openai import OpenAI
import json
import streamlit as st

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
    st.error(f"Error authenticating with Reddit API: {str(e)}")
    st.error("Please check your Reddit API credentials in the .env file.")
    st.stop()
except prawcore.exceptions.OAuthException as e:
    st.error(f"OAuth Error: {str(e)}")
    st.error("Please verify your Reddit API credentials and ensure they have the correct permissions.")
    st.stop()

# Set up OpenAI client
client = OpenAI(api_key=get_env_variable("OPENAI_API_KEY"))

def search_reddit_posts(query, limit, time_filter, sort, subreddit=None):
    posts = []
    limit = min(limit, 50)  # Ensure limit is at most 50

    if subreddit:
        search_subreddit = reddit.subreddit(subreddit)
    else:
        search_subreddit = reddit.subreddit("all")

    if sort == "relevance":
        submissions = search_subreddit.search(query, sort=sort, time_filter=time_filter, limit=limit)
    elif sort == "hot":
        submissions = search_subreddit.search(query, sort="hot", limit=limit)
    elif sort == "new":
        submissions = search_subreddit.search(query, sort="new", limit=limit)
    elif sort == "top":
        submissions = search_subreddit.search(query, sort="top", time_filter=time_filter, limit=limit)
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

    # Sort posts by score in descending order
    posts.sort(key=lambda x: x['score'], reverse=True)
    return posts

def get_sentiment(text):
    response = client.chat.completions.create(
        model="gpt-4o",
        messages=[
            {"role": "system", "content": "You are a sentiment analysis tool. Respond with one word describing the primary emotion in the text. Examples include: positive, negative, neutral, angry, depressed, sad, excited, happy, anxious, etc."},
            {"role": "user", "content": f"Analyze the primary emotion in this text: {text}"}
        ],
        max_tokens=1
    )
    sentiment = response.choices[0].message.content.strip().lower()
    return sentiment

def filter_reddit_posts(posts, sentiment=None, max_posts=10):
    filtered_posts = []
    for post in posts:
        post_sentiment = get_sentiment(post['title'])
        post['sentiment'] = post_sentiment
        
        if sentiment is None or post['sentiment'] == sentiment:
            filtered_posts.append(post)
        
        if len(filtered_posts) == max_posts:
            break
    
    return filtered_posts

def analyze_reddit_posts(user_prompt):
    functions = [
        {
            "name": "search_reddit_posts",
            "description": "Search for max 50 posts across all subreddits or a specific subreddit, ordered by score (upvotes)",
            "parameters": {
                "type": "object",
                "properties": {
                    "query": {
                        "type": "string",
                        "description": "Search query to find relevant posts"
                    },
                    "limit": {
                        "type": "integer",
                        "description": "Maximum number of posts to fetch (max 50)"
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
        },
        {
            "name": "filter_reddit_posts",
            "description": "Filter Reddit posts based on sentiment and limit the number of results",
            "parameters": {
                "type": "object",
                "properties": {
                    "posts": {
                        "type": "array",
                        "items": {
                            "type": "object"
                        },
                        "description": "List of Reddit posts to filter"
                    },
                    "sentiment": {
                        "type": "string",
                        "description": "Sentiment to filter by (optional). Can be any emotion like positive, negative, neutral, angry, depressed, sad, excited, happy, anxious, etc."
                    },
                    "max_posts": {
                        "type": "integer",
                        "description": "Maximum number of posts to return (default 10)"
                    }
                },
                "required": ["posts"]
            }
        }
    ]

    messages = [
        {"role": "system", "content": "You are a helpful assistant that analyzes Reddit posts. Use the search_reddit_posts function to search for relevant posts across all subreddits or a specific subreddit based on the user's request. Then use the filter_reddit_posts function to apply sentiment analysis and limit the results. Always include the URL, subreddit name, created date and score for each Reddit post in your analysis. If the user specifies a subreddit, make sure to include it in the function call. The sentiment analysis now includes a wider range of emotions."},
        {"role": "user", "content": user_prompt}
    ]

    response = client.chat.completions.create(
        model="gpt-4o",
        messages=messages,
        functions=functions,
        function_call="auto",
        stream=True
    )

    collected_messages = []
    function_name = None
    function_args = None

    for chunk in response:
        if chunk.choices[0].delta.function_call:
            if function_name is None:
                function_name = chunk.choices[0].delta.function_call.name
            if chunk.choices[0].delta.function_call.arguments:
                collected_messages.append(chunk.choices[0].delta.function_call.arguments)
        elif chunk.choices[0].delta.content:
            yield chunk.choices[0].delta.content

    if function_name:
        function_args = json.loads(''.join(collected_messages))
        if function_name == "search_reddit_posts":
            posts = search_reddit_posts(**function_args)
            
            messages.append({
                "role": "function",
                "name": function_name,
                "content": json.dumps(posts)
            })
            
            filter_response = client.chat.completions.create(
                model="gpt-4o-mini",
                messages=messages,
                functions=functions,
                function_call={"name": "filter_reddit_posts"},
                stream=True
            )
            
            filter_collected_messages = []
            for chunk in filter_response:
                if chunk.choices[0].delta.function_call and chunk.choices[0].delta.function_call.arguments:
                    filter_collected_messages.append(chunk.choices[0].delta.function_call.arguments)
            
            filter_args = json.loads(''.join(filter_collected_messages))
            filtered_posts = filter_reddit_posts(posts, filter_args.get('sentiment'), filter_args.get('max_posts', 10))
            
            messages.append({
                "role": "function",
                "name": "filter_reddit_posts",
                "content": json.dumps(filtered_posts)
            })
            
            final_response = client.chat.completions.create(
                model="gpt-4o-mini",
                messages=messages,
                stream=True
            )
            
            for chunk in final_response:
                if chunk.choices[0].delta.content:
                    yield chunk.choices[0].delta.content

def main():
    st.set_page_config(page_title="Reddit GPT")
    st.title("Reddit GPT")

    user_prompt = st.text_input("Put r/... to get posts from a specific subreddit, or leave specificities out to search across reddit", placeholder="what are the hottest posts about Elon Musk now?")
    
    if user_prompt:
        with st.spinner("Analyzing Reddit posts..."):
            # Create a placeholder for the streamed output
            analysis_placeholder = st.empty()
            
            # Stream the analysis
            full_response = ""
            for response_chunk in analyze_reddit_posts(user_prompt):
                full_response += response_chunk
                analysis_placeholder.markdown(full_response)

if __name__ == "__main__":
    main()
