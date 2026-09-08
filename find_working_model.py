import os, requests, dotenv
dotenv.load_dotenv()

api_key = os.getenv('IBM_CLOUD_API_KEY')
project_id = os.getenv('WATSONX_PROJECT_ID')
url = os.getenv('WATSONX_URL')

token = requests.post(
    'https://iam.cloud.ibm.com/identity/token',
    data={'grant_type': 'urn:ibm:params:oauth:grant-type:apikey', 'apikey': api_key}
).json().get('access_token')

headers = {'Authorization': f'Bearer {token}', 'Content-Type': 'application/json'}

specs = requests.get(f'{url}/ml/v1/foundation_model_specs?version=2023-05-29', headers=headers).json()

# Filter strictly for models whose specs support the text_generation function
candidates = [
    m['model_id'] for m in specs.get('resources', [])
    if any(f.get('id') == 'text_generation' for f in m.get('functions', []))
]

print(f'Testing {len(candidates)} models supporting text_generation...')
for model in candidates:
    payload = {
        'input': 'Test prompt',
        'model_id': model,
        'project_id': project_id,
        'parameters': {'max_new_tokens': 1}
    }
    res = requests.post(f'{url}/ml/v1/text/generation?version=2023-05-29', headers=headers, json=payload)
    if res.status_code == 200:
        print(f'\n--> VERIFIED WORKING: MODEL_ID={model}')
        break
    else:
        err = res.json().get('errors', [{}])[0].get('code', 'unknown')
        print(f'[{res.status_code} {err}] {model}')
