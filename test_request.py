import requests

resp = requests.post("http://localhost:5000/predict",
                     files={"file": open('images/training/ripe/banana-ripe-8.png','rb')})

print(resp.json())
