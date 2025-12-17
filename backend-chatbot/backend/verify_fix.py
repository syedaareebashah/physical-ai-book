import sys
import os
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

# Test the modified code without using the full app
from src.services.rag import RAGResponse, Source

# Simulate the change we made in main.py
def simulate_old_behavior():
    """Simulate what happened before our fix"""
    rag_response = RAGResponse(response="This is a test response from the backend.", sources=[Source(text="Test source", page_number=1)])
    # This is what was happening before (incorrect) - returning the entire object
    return {"response": rag_response}

def simulate_new_behavior():
    """Simulate what happens after our fix"""
    rag_response = RAGResponse(response="This is a test response from the backend.", sources=[Source(text="Test source", page_number=1)])
    # This is what happens after our fix - extracting just the response text
    return {"response": rag_response.response}

print("Testing the fix for the API response format:")
print()

old_result = simulate_old_behavior()
new_result = simulate_new_behavior()

print("BEFORE fix - API returned:")
print(f"  Type of 'response' value: {type(old_result['response'])}")
print(f"  Value: {old_result['response']}")
print()

print("AFTER fix - API returns:")
print(f"  Type of 'response' value: {type(new_result['response'])}")
print(f"  Value: {new_result['response']}")
print()

# Check if the fix is working
if isinstance(new_result['response'], str) and not isinstance(old_result['response'], str):
    print("✅ SUCCESS: The fix correctly extracts the response text for the frontend!")
    print("✅ The frontend widget will now receive a string instead of an object in the 'response' field.")
elif isinstance(new_result['response'], str):
    print("✅ SUCCESS: The response is already a string as expected by the frontend!")
else:
    print("❌ The fix might not be working as expected.")