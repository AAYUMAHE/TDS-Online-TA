import requests
import json
import time
from datetime import datetime

BASE_URL = "https://discourse.onlinedegree.iitm.ac.in"
CATEGORY_ID = 34
DISCOURSE_SESSION_TOKEN = 'wAIb74L66iBtoqf0%2BLHgHdkA2JBDwFmQmnhb%2BoaZ9UPJh3K12Un4iJ4cimqYj8Z2W2A117kIgFzzfW4Ct6csVK1J%2FvqRPKmPxVKT%2Fx92YofdcaCg4mcZEQx5e%2FkxoQkoUq%2FCcI%2BVR7N1S1ym%2FV0pwrakdXwomwVQDJp%2BeFU7%2BuYkabxj1fCViNNMe3UuEDkcNqksrAsgXP9JFjEfAY3U1z%2F2EolfCO466%2BprfSfmpohSiAJF5FapNfpQTFdrY1Ld3Vfl2L844jKnBbrpIC%2BpJ1CfoEw0WRvOnzm5hHvq0ZaGrDPcXW3M2I%2BRIxs1znKX--vGjwhx9F7%2FwAUuVZ--3xCJYwZ8cUMnLV5ycu1Xig%3D%3D'

session = requests.Session()
session.cookies.set("_t", DISCOURSE_SESSION_TOKEN, domain="discourse.onlinedegree.iitm.ac.in")

HEADERS = {
    "User-Agent": "Mozilla/5.0"
}

# === SET YOUR CUTOFF DATE HERE ===
DATE_FILTER = "01/01/2024"  # dd/mm/yyyy
DATE_CUTOFF = datetime.strptime(DATE_FILTER, "%d/%m/%Y")

def get_category_topics(category_id, page_limit=300):
    """Fetch topic IDs from a Discourse category after a specific date."""
    topics = []
    for page in range(0, page_limit):
        url = f"{BASE_URL}/c/courses/tds-kb/{category_id}.json?page={page}"
        print(f"Fetching category page: {url}")
        res = session.get(url, headers=HEADERS)
        if res.status_code != 200:
            print("Failed to fetch category page:", res.status_code)
            break
        data = res.json()
        page_topics = data.get("topic_list", {}).get("topics", [])
        if not page_topics:
            break

        for topic in page_topics:
            # Use `last_posted_at` or `created_at` if available
            date_str = topic.get("last_posted_at") or topic.get("created_at")
            if date_str:
                topic_datetime = datetime.strptime(date_str, "%Y-%m-%dT%H:%M:%S.%fZ")
                if topic_datetime > DATE_CUTOFF:
                    topics.append(topic)

        time.sleep(1)  # Be polite
    return topics

def fetch_topic_posts(topic_id):
    """Fetch all posts from a topic (thread)."""
    url = f"{BASE_URL}/t/{topic_id}.json"
    res = session.get(url, headers=HEADERS)
    if res.status_code != 200:
        print(f"Failed to fetch topic {topic_id}")
        return None
    return res.json()

def main():
    all_data = []
    topics = get_category_topics(CATEGORY_ID, page_limit=5)
    print(f"Found {len(topics)} topics after {DATE_FILTER}.")

    for topic in topics:
        topic_id = topic["id"]
        print(f"Fetching topic ID {topic_id} - {topic['title']}")
        thread_data = fetch_topic_posts(topic_id)
        if thread_data:
            all_data.append(thread_data)
        time.sleep(1)

    with open("tds_kb_filtered_threads_by_date.json", "w", encoding="utf-8") as f:
        json.dump(all_data, f, indent=2, ensure_ascii=False)
    print("Filtered threads saved to tds_kb_filtered_threads.json")

if __name__ == "__main__":
    main()
