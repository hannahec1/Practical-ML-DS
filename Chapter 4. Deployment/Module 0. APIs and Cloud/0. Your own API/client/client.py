import requests
import json

r = requests.get('http://127.0.0.1:5000') # make a get request to the local host (look in your hosts file)
# https://superuser.com/questions/949428/whats-the-difference-between-127-0-0-1-and-0-0-0-0

print(r.__dir__()) # 
print(r.json()) # get the bytes back

r = requests.post('http://127.0.0.1:5000/predict', data=json.dumps({'input': 3}), headers={'content-type': 'application/json'})
print(r.content)