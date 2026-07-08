import requests
import json

# 1. Define the API URL
# Replace with your actual server address (e.g., http://localhost:8000/sam)
API_URL = "http://localhost:8111/sam"

# 2. Prepare the payload matching the QueryRequest model
payload = {
    "prompt": "fish",  # Replace with your actual prompt
    "image_paths": [                           # List of image paths/URLs
        "/project/marieninst/an/marineevt/test/CasualReasoning/ReasonInference/videos/FJHQoRoh7vhV/frames/frame_000000.jpg",
    ],
    "ground_type": "highest",             # Replace with your actual ground_type (e.g., "mask", "box")
}

try:
    # 3. Make the POST request
    response = requests.post(API_URL, json=payload)
    
    # 4. Check for HTTP errors
    response.raise_for_status()
    
    # 5. Parse and print the response
    result = response.json()
    print("Success!")
    print(json.dumps(result, indent=4))

except requests.exceptions.HTTPError as http_err:
    print(f"HTTP error occurred: {http_err} - Response: {response.text}")
except Exception as err:
    print(f"An error occurred: {err}")