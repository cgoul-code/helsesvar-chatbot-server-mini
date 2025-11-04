## Llama Index Server Application

A server application demonstrating Retrieval-Augmented Generation (RAG) with Azure OpenAI and LlamaIndex, powered by LangChain. This project, developed by **Helsedirektoratet** as part of the **HelseSvar** initiative, provides a chat-like server interface for querying and retrieving indexed health articles.

---

### Features

- **Azure OpenAI Integration**: Utilizes Azure OpenAI models for advanced language understanding.
- **Vector Indexing**: Stores and retrieves text data using LlamaIndex.
- **Natural Language Queries**: Supports conversational queries over indexed data.
- **Context-Aware Responses**: Generates answers based on search results.
- **Agent Workflow**: Orchestrates multi-step reasoning for structured answers.
- **Support markdowns**: Answers include markdown markup 

---

### Prerequisites

- **Python**: Version 3.13 or newer
- **Quart**: ASGI web framework for Python
- **Azure OpenAI Service**: An active resource with deployment name, API key, endpoint, and API version

---

### Installation

1. **Clone the repository**:
    ```bash
    git clone <repository_url>
    cd <repository_folder>
    ```

2. **Create a virtual environment** (optional but recommended):
    ```bash
    python -m venv .venv
    source .venv/bin/activate  # On Windows: .\.venv\Scripts\activate
    ```

3. **Install dependencies**:
    ```bash
    pip install -r requirements.txt
    ```

---

### Configuration

1. **Create a `.env` file** in the project root with the following keys:
    ```dotenv
    AZURE_OPENAI_MODEL=<your_model_name>
    AZURE_OPENAI_DEPLOYMENT_NAME=<your_deployment>
    AZURE_OPENAI_API_KEY=<your_api_key>
    AZURE_OPENAI_AZURE_ENDPOINT=<your_endpoint>
    AZURE_OPENAI_API_VERSION=<your_api_version>
    OPENAI_API_KEY=<your_openai_key>  # Optional if using Azure only
    ```

2. **Data Indexing**
   - By default, the application indexes health articles from the folder:
   ```
   blobstorage/chatbot/helsenorgeartikler
   ```
   - The repository includes a prebuilt vector index of 1,853 pages from Helsenorge.no.
   - unzip the file helsenorgeartikler.zip in the same folder

---


### Project Structure

```

   ├── .env                                     # Secret keys
   ├── requirements.txt
   ├── app.py                                   # Main application entry point
   ├── config.py                                # Server configuration loader
   ├── answer_utils.py                          # Query handling and answer generation
   ├── query_utils.py                           # Query parsing utilities
   ├── routes/                                  # HTTP route definitions
   ├── agent_workflow_structured_answer.py      # Agent workflow implementation
   └── blobstorage/chatbot/helsenorgeartikler   # Indexed article data
```

---

### Agent Workflow Visualization

![Agent Workflow](./docs/agent_workflow_structured_answer.png)



### Running Locally

Start the server on port 80:

```bash
python app.py
```

Open your browser at `http://localhost:80` to interact with the chat interface.

---

### Testing with Postman

#### Download and install Postman:

- Visit https://www.postman.com/downloads/ and select the appropriate installer for your OS.

- Follow the installation prompts.

#### Create a new request:

- Open Postman and click New > HTTP Request.

- Set the method to POST and the URL to http://127.0.0.1:80/chat.

#### Configure the request body:

- Select Body > raw and choose JSON from the dropdown.


#### Paste the following JSON payload:
```bash

{
  "messages": [
    {
      "role": "user",
      "content": "Hvordan fungerer en hjertestarter?"
    }
  ],
  "similarity_top_k": "10",      # how many documents from the vectorindex to work with
  "similarity_cutoff": "0.75",   # index for simularity
  "vectorIndex": "helsenorgeartikler",  # vectorindex to work with
  "agent": "agent_workflow_structured_answer", # agent used for answering the query
}
```

#### Send the request:

- Click Send.

- Inspect the response in the Postman console to verify the server's output. It shoud respond something like:
``` bash
{
    "answer": "# Oppsummering av spørsmålet\n\n## Spørsmålet fra brukeren\nHvordan fungerer en hjertestarter?\n\n## Tittel\nHvordan jeg bruker en hjertestarter?\n\n## Kort sammendrag av spørsmålet\nBrukeren spør om hvordan en hjertestarter fungerer.\n\n## Lettlest svar\nEn hjertestarter brukes ved hjertestans sammen med hjerte- og lungeredning (HLR). Når noen opplever hjertestans, kan en hjertestarter bidra til å gjenopprette en normal hjerterytme. \n\nSlik fungerer en hjertestarter:\n1. En hjertestarter brukes ved hjertestans. Den har to elektroder som festes på brystet til personen i hjertestans.\n2. Ved hjelp av disse elektrodene kan hjertestarteren gi ett eller flere strømstøt gjennom hjertet. Dette gir hjertet en sjanse til å starte på nytt med riktig rytme.\n3. Hjertestarteren analyserer hjerterytmen og gir instruksjoner om hva du skal gjøre. Hvis strømstøt ikke vil ha effekt, vil hjertestarteren be deg fortsette med hjerte- og lungeredning.\n4. Hjertestarteren vil også be deg om å stoppe HLR for å analysere hjerterytmen på nytt hvert tredje minutt.\n5. Dersom personen har ventrikkelflimmer eller ventrikkeltakykardi (kaotiske hjerterytmer), vil hjertestarteren anbefale å gi strømstøt ved å trykke på sjokknappen.\n6. Etter at strømstøt er gitt, må du fortsette med HLR inntil ambulansen kommer eller til du ser sikre tegn til liv.\n\nDet er viktig å følge hjertestarterens instruksjoner nøye og fortsette med hjerte- og lungeredning til ambulansepersonell overtar eller til personen viser tegn til liv.\n\n## Referanser\n- [Slik bruker du hjertestarter - Helsenorge](https://www.helsenorge.no/forstehjelp/bruk-av-hjertestarter/) (Relevans: 0.89)\n- [Slik bruker du hjertestarter - Helsenorge](https://www.helsenorge.no/forstehjelp/bruk-av-hjertestarter/) (Relevans: 0.88)\n- [Phillips Heart Start 1 - Helsenorge](https://www.helsenorge.no/forstehjelp/registrer-din-hjertestarter/phillips-heart-start-1/) (Relevans: 0.85)\n- [Registrer din hjertestarter - Helsenorge](https://www.helsenorge.no/forstehjelp/registrer-din-hjertestarter/) (Relevans: 0.85)\n- [Phillips Heartstart FRX - Helsenorge](https://www.helsenorge.no/forstehjelp/registrer-din-hjertestarter/phillips-heartstart-frx/) (Relevans: 0.85)\n- [Philips HeartStart FR3 - Helsenorge](https://www.helsenorge.no/forstehjelp/registrer-din-hjertestarter/philips-heartstart-fr3/) (Relevans: 0.84)\n- [Hjartestans – Ring 113 og start hjartekompresjonar - Helsenorge](https://www.helsenorge.no/sykdom/hjerte-og-kar/hjertestans/) (Relevans: 0.84)\n- [Hjerte- og lungeredning (HLR) - slik gir du livreddende førstehjelp - Helsenorge](https://www.helsenorge.no/forstehjelp/hjerte-og-lungeredning/) (Relevans: 0.84)\n- [Hjartestans – Ring 113 og start hjartekompresjonar - Helsenorge](https://www.helsenorge.no/sykdom/hjerte-og-kar/hjertestans/) (Relevans: 0.83)\n- [Primedic Heartsave ONE - Helsenorge](https://www.helsenorge.no/forstehjelp/registrer-din-hjertestarter/primedic-heartsave-one/) (Relevans: 0.83)\n"
}
```
![Postman example](./docs/postman.png)

---

### Deployment to Azure

1. **Create a Web App** on Azure with Python 3.13 runtime.
2. **General Settings**:
   - **Startup Command**: `bash startup.sh`
   - **SCM/FTPS Basic Auth**: Enabled
   - **Path Mapping**: `mountName=chatbot`, `mountPath=/blobstorage/chatbot`
3. **Deployment Center**:
   - Connect to your GitHub repo for continuous deployment.
4. **Blob Storage**:
   - Create a storage account and container named `chatbot`.
   - Upload the `helsenorgeartikler` folder to `chatbot` container.

---

Now you’re ready to explore health-related queries with RAG and agent-driven workflows!


### Technical developper tips
## requirements.txt
pip freeze >> requirements.txt
pip install -r requirements.txt
remember to remove pywin32==306 from requirements.txt
pywinpty==3.0.0; sys_platform == "win32"

## to run build script for vectorindex (as a module):
python -m utils.create_vectorindex.build_hei_index


## reinstall 
# 0) Close any Python terminals/VSCode tasks using the venv

# 1) Delete the existing venv
Remove-Item -Recurse -Force .\.venv

# 2) Recreate & activate
python -m venv .venv 
& .\.venv\Scripts\Activate.ps1

# 3) Upgrade build tooling
python -m pip install -U pip wheel setuptools

# 4) Clear the pip cache (user-level, outside venv)
pip cache dir
pip cache purge
# If 'pip cache purge' isn’t available, delete the folder shown by "pip cache dir":
# Remove-Item -Recurse -Force "$env:LocalAppData\pip\Cache"

# 5) Fresh install from requirements.txt with no cache
.\.venv\Scripts\python.exe -m pip install --no-cache-dir -r requirements.txt

## Tips for reinstall llama-index:
pip install llama-index --upgrade --no-cache-dir --force-reinstall 
pip check    

:
git config --edit

start server: hypercorn server:app --bind 0.0.0.0:80

azure CLI:
(i powershell: az login --tenant helsedir.no)

to access /Home dir:
az webapp ssh --name chatbot3-llama-server --resource-group rg-webappaichatbot

To access logging:
az webapp log config -g rg-webappaichatbot -n helsesvar-chatbot-server-mini --web-server-logging filesystem
az webapp log tail  -g rg-webappaichatbot -n helsesvar-chatbot-server-mini

cd /home/logs
cat requests.log.2025-05-2

Download the containers:
 az storage blob download-batch --destination C:/Users/ChristianGoulignac/Downloads --source chatbot --account-name helsesvar --account-key COzcrGBg34QJ+KF1yAXG71xuIlbzFIDk2MxjH7If7Dfa22vhU5CftWhqrqFx+u10Pa+B85ryt3NR+AStEIpHLA==

## kjøre build index under utils:
# C:\Users\ChristianGoulignac\vscode\llama
.\llama_chatbot_server_mini\.venv\Scripts\python.exe -m llama_chatbot_server_mini.utils.create_vectorindex.build_hei_index

## Kjøre testene
python -m test.data.batch_answer_to_excel --input test/data/sample_questions.json --out test/data/answers.xlsx --vector-index hvaerinnafor