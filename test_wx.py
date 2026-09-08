import os, requests, dotenv
dotenv.load_dotenv()
token = requests.post('https://iam.cloud.ibm.com/identity/token', data={'grant_type': 'urn:ibm:params:oauth:grant-type:apikey', 'apikey': os.getenv('IBM_CLOUD_API_KEY')}).json().get('access_token')
r = requests.post(f"{os.getenv('WATSONX_URL')}/ml/v1/text/generation?version=2023-05-29", headers={'Authorization': f'Bearer {token}', 'Content-Type': 'application/json'}, json={'input': 'Hello', 'model_id': os.getenv('MODEL_ID'), 'project_id': os.getenv('WATSONX_PROJECT_ID')})
print('Status:', r.status_code)
print('Response:', r.text)
