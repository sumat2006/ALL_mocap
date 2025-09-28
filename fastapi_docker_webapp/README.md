# FastAPI Docker Web App

Build and run with Docker Compose (from project root):

```bash
docker compose up --build
```


Application will be available at `http://localhost:8000/`.

- `GET /` : interactive page to send text to `/predict`
- `POST /predict` : JSON API, body `{ "text": "..." }`
- `POST /predict-form` : Form POST (used by the interactive page)
- `GET /logs` : web log viewer (last 200 lines by default) 

Logs are written to `app/logs/app.log` and rotated.



# FastAPI roun with Uvicorn

```bash
pip install uvicorn
```

```bash
uvicorn main:app --host 0.0.0.0 --port 80 --reload
```
to run with uvicorn

# Basic Knowledge

| Model Type | Model Name | Link |
| :------ | :---------- | :------ |
| ASR | Thonburain whisper | https://github.com/biodatlab/thonburian-whisper | 
| LLM | KhanommThan | https://huggingface.co/pythainlp/KhanomTanLLM-1B |
| TTS | vachanatts | - |
| Hand-Sign | Our model | - |

*all of the model will be kept in app/asset/model. Template can be found in app/static/index.html and css, javascript is there also.* 




```
|── asset
  ├── Rollback.json                 # transform number from model prediction to text
  ├── config.yaml                   # path configuration
  ├── utils.py                      # store function
  |── model                         # store all model
```

Example of config.yaml
```yaml
seq_name: ./asset/model/KhanomTanLLM-1B
shl_path: ./asset/model/model_89.pth
ft_model: ./asset/model/thai_tts/model/model_850000.pt
vocab_file: ./asset/model/thai_tts/vocab.txt
rollback: ./asset/rollback.json
asr_path: ./asset/model/thonburain-whisper
```


Template (function explain is in /app/README.md)
```
|── static
  ├── index.html                  # base template html
  ├── logs.html                   # store logs
  ├── script.js                   # store list of javascript function
  ├── styles.css                  # store css
```

## Vanchantts
> is use to do tts in this work.

# Great Reminder
  The fall of accuracy occured from a testset that is less than we expect and need to test and data is not in a realword situation enough so the accuracy will be falling aprat when test on production in this case 89 -> 43 acc.
  Accuracy will be depended on datatype and model type and model size. Speed will be depended on comupute unit (GPU)

  Language model , TTS , ASR still lack of speed due to not enough compute unit. 


  if you want to conitnue in this project please update GPU , plans and data first and every thing can keep continue easier

  
