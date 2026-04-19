import base64
import urllib.request
import json

with open('training_and_resources/test.jpg', 'rb') as f:
    b64 = base64.b64encode(f.read()).decode('utf-8')

payload = json.dumps({'image': b64}).encode('utf-8')
req = urllib.request.Request('http://localhost:8000/predict', data=payload, headers={'Content-Type': 'application/json'})

try:
    with urllib.request.urlopen(req) as response:
        print('Status:', response.getcode())
        print('Response:', response.read().decode('utf-8'))
except urllib.error.HTTPError as e:
    print('HTTPError:', e.code)
    print('Error Body:', e.read().decode('utf-8'))
except Exception as e:
    print('Exception:', str(e))