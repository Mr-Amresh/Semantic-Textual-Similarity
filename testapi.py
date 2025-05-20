import requests

url = "https://dataneuron-project-366741234981.us-central1.run.app/similarity"
test_cases = [
    {
        "text1": "broadband challenges tv viewing the number of europeans with broadband has exploded over the past 12 months...",
        "text2": "gardener wins double in glasgow britain s jason gardener enjoyed a double 60m success..."
    },
    {
        "text1": "12345767899000",
        "text2": "gardener wins double in glasgow britain s jason gardener enjoyed a double 60m success..."
    },
    {
        "text1": "",
        "text2": "text"
    },
    {
        "text1": "a" * 11000,
        "text2": "text"
    }
]

for i, case in enumerate(test_cases, 1):
    try:
        response = requests.post(url, json=case, timeout=10)
        print(f"Test {i}: Status {response.status_code}")
        print(f"Headers: {response.headers}")
        print(f"Raw Response: {response.text[:500]}")  # Truncate for readability
        try:
            print(f"JSON Response: {response.json()}")
        except ValueError as e:
            print(f"JSON Parse Error: {e}")
    except requests.RequestException as e:
        print(f"Test {i}: Failed - {e}")