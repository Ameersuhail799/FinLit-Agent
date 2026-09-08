import os, requests, dotenv
dotenv.load_dotenv()
token = requests.post('https://iam.cloud.ibm.com/identity/token', data={'grant_type': 'urn:ibm:params:oauth:grant-type:apikey', 'apikey': os.getenv('IBM_CLOUD_API_KEY')}).json().get('access_token')
r = requests.get(f"{os.getenv('WATSONX_URL')}/ml/v1/foundation_model_specs?version=2023-05-29", headers={'Authorization': f'Bearer {token}'})
models = [m['model_id'] for m in r.json().get('resources', []) if 'granite' in m['model_id']]
print('Supported Granite models in your Dallas instance:\n', '\n'.join(models))
