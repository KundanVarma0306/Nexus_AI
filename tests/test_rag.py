import requests
import time
import json
import os

BASE_URL = "http://localhost:8000"

def test_health():
    print("Testing /health...")
    response = requests.get(f"{BASE_URL}/health")
    assert response.status_code == 200
    data = response.json()
    assert data["status"] == "healthy"
    print("✅ Health check passed")

def test_upload():
    print("Testing /api/upload...")
    with open("test_doc.txt", "w") as f:
        f.write("DeepMind is an AI research laboratory based in London. Its mission is to 'solve intelligence' and use it to solve everything else.")
    
    with open("test_doc.txt", "rb") as f:
        response = requests.post(f"{BASE_URL}/api/upload", files={"file": f})
    
    assert response.status_code == 200
    data = response.json()
    assert data["success"] is True
    print("✅ Upload passed")
    os.remove("test_doc.txt")

def test_query():
    print("Testing /api/query...")
    query_data = {
        "query": "Where is DeepMind based?",
        "top_k": 3
    }
    response = requests.post(f"{BASE_URL}/api/query", json=query_data)
    assert response.status_code == 200
    data = response.json()
    assert "London" in data["answer"]
    assert len(data["sources"]) > 0
    print("✅ Query passed")

def test_history():
    print("Testing /api/history...")
    response = requests.get(f"{BASE_URL}/api/history")
    assert response.status_code == 200
    data = response.json()
    assert len(data) > 0
    print("✅ History fetch passed")

if __name__ == "__main__":
    try:
        test_health()
        test_upload()
        time.sleep(2) # Give Chroma some time to settle
        test_query()
        test_history()
        print("\n🚀 ALL TESTS PASSED SUCCESSFULLY!")
    except Exception as e:
        print(f"\n❌ TEST FAILED: {e}")
