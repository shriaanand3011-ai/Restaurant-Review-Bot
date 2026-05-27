import os
import requests
from google import genai
from flask import Flask, request, jsonify, render_template
from dotenv import load_dotenv

# Load secret keys from .env file
load_dotenv()
PLACES_KEY = os.getenv("GOOGLE_PLACES_API_KEY")
GEMINI_KEY = os.getenv("GEMINI_API_KEY")

# Set up Gemini
client = genai.Client(api_key=GEMINI_KEY)

# Set up the web server
app = Flask(__name__)


def find_place(restaurant_name):
    """Step 1: Find the restaurant on Google Maps and get its place_id"""
    url = "https://maps.googleapis.com/maps/api/place/findplacefromtext/json"
    params = {
        "input": restaurant_name,
        "inputtype": "textquery",
        "fields": "place_id,name,formatted_address",
        "key": PLACES_KEY
    }
    response = requests.get(url, params=params).json()
    candidates = response.get("candidates", [])
    if not candidates:
        return None
    return candidates[0]


def get_reviews(place_id):
    """Step 2: Use the place_id to fetch reviews"""
    url = "https://maps.googleapis.com/maps/api/place/details/json"
    params = {
        "place_id": place_id,
        "fields": "name,rating,user_ratings_total,reviews,formatted_address,types,editorial_summary",
        "key": PLACES_KEY
    }
    response = requests.get(url, params=params).json()
    return response.get("result", {})


def analyze_with_gemini(restaurant_name, reviews_data):
    """Step 3: Send reviews to Gemini and ask for summary + cuisine + top dishes"""
    reviews = reviews_data.get("reviews", [])
    if not reviews:
        return "No reviews found for this place."

    # Extract cuisine hints from Google's data
    google_types = reviews_data.get("types", [])
    editorial = reviews_data.get("editorial_summary", {}).get("overview", "")

    # Filter out generic types like "point_of_interest", "establishment", "food"
    skip_types = {"point_of_interest", "establishment", "food", "restaurant",
                  "cafe", "bar", "bakery", "meal_takeaway", "meal_delivery",
                  "store", "business"}
    cuisine_hints = [t.replace("_", " ") for t in google_types if t not in skip_types]
    google_hint = ", ".join(cuisine_hints) if cuisine_hints else "not specified"

    # Stitch reviews into one big text block
    review_text = "\n\n".join([
        f"Rating: {r.get('rating')}/5\nReview: {r.get('text', '')}"
        for r in reviews
    ])

    prompt = f"""You are a food guide assistant. Analyze these Google Maps reviews for "{restaurant_name}" and reply in this exact format with plain text (no markdown, no asterisks):

CUISINE
[1-2 lines naming the cuisine(s) served, e.g. "North Indian, Mughlai" or "Italian & wood-fired pizza". Use Google's hint plus what reviewers actually mention.]

OVERALL VIBE
[2-3 sentences capturing both positive and negative themes]

WHAT PEOPLE LOVE
- [point 1]
- [point 2]
- [point 3]

COMMON COMPLAINTS
- [point 1]
- [point 2]

TOP 5 DISHES TO TRY
1. [dish name] - [why reviewers recommend it]
2. [dish name] - [why]
3. [dish name] - [why]
4. [dish name] - [why]
5. [dish name] - [why]

If reviewers don't mention 5 specific dishes, list fewer and say "Reviews don't mention enough specific dishes" for the rest.

Google's category hint: {google_hint}
{f'Editorial description: {editorial}' if editorial else ''}

Here are the reviews:

{review_text}"""

    response = client.models.generate_content(
        model="gemini-flash-latest",
        contents=prompt
    )
    return response.text



@app.route("/")
def home():
    return render_template("index.html")

@app.route("/autocomplete", methods=["POST"])
def autocomplete():
    data = request.get_json()
    query = data.get("query", "").strip()
    lat = data.get("lat")
    lng = data.get("lng")

    if not query or len(query) < 2:
        return jsonify({"predictions": []})

    url = "https://maps.googleapis.com/maps/api/place/autocomplete/json"
    params = {
        "input": query,
        "key": PLACES_KEY,
        # Restrict to food-related places
        "types": "restaurant|cafe|bakery|bar|food",
    }

    # If user gave location, bias results to nearby (50km radius)
    if lat and lng:
        params["location"] = f"{lat},{lng}"
        params["radius"] = 50000

    response = requests.get(url, params=params).json()
    predictions = response.get("predictions", [])

    # Simplify the response — only send what frontend needs
    results = [
        {
            "place_id": p["place_id"],
            "main_text": p.get("structured_formatting", {}).get("main_text", ""),
            "secondary_text": p.get("structured_formatting", {}).get("secondary_text", ""),
        }
        for p in predictions
    ]

    return jsonify({"predictions": results})


@app.route("/search", methods=["POST"])
def search():
    data = request.get_json()
    restaurant_name = data.get("name", "").strip()
    place_id = data.get("place_id", "").strip()

    if not restaurant_name and not place_id:
        return jsonify({"error": "Please enter a restaurant name"}), 400

    # If user clicked a dropdown suggestion, use that place_id directly
    if not place_id:
        place = find_place(restaurant_name)
        if not place:
            return jsonify({"error": f"Couldn't find '{restaurant_name}' on Google Maps. Try a more specific name."}), 404
        place_id = place["place_id"]

    # Get reviews
    details = get_reviews(place_id)

    # Summarize
    analysis = analyze_with_gemini(details.get("name", restaurant_name), details)

    # Pull cuisine tags for the frontend chip
    skip_types = {"point_of_interest", "establishment", "food", "restaurant",
                  "cafe", "bar", "bakery", "meal_takeaway", "meal_delivery",
                  "store", "business"}
    cuisine_tags = [t.replace("_", " ").title()
                    for t in details.get("types", [])
                    if t not in skip_types]

    return jsonify({
        "name": details.get("name"),
        "address": details.get("formatted_address"),
        "rating": details.get("rating"),
        "total_reviews": details.get("user_ratings_total"),
        "cuisine_tags": cuisine_tags,
        "analysis": analysis
    })


if __name__ == "__main__":
    port = int(os.environ.get("PORT", 5000))
    app.run(host="0.0.0.0", port=port, debug=False)