from fastapi import FastAPI, Request, Form,UploadFile, File,WebSocketDisconnect,WebSocket,HTTPException
from fastapi.responses import HTMLResponse, JSONResponse,StreamingResponse
from fastapi.staticfiles import StaticFiles
from fastapi.templating import Jinja2Templates
from typing import List, Optional , Annotated
from pydantic import BaseModel
from typing import Optional
from importlib.resources import files
import logging
from logging.handlers import RotatingFileHandler
import os
print(os.curdir)
print(os.listdir("./"))
from pathlib import Path
from asset.utils import S2S,RapidChangeWindowClassifier,CNNTimeSeriesClassifier,Thonburain
import base64
import torch
import json
import yaml
import soundfile as sf
import io
from pydub import AudioSegment
import numpy as np
from vachanatts import TTS
from collections import deque
import soundfile as sf
import pandas as pd
import time
"-----------------------------------------"
### initial path
with open("./asset/config.yaml", 'r') as file:
        config = yaml.load(file, Loader=yaml.FullLoader)
        print("YAML data loaded successfully:")
seq_name = config["seq_name"]
shl_path = config["shl_path"]
ft_model = config["ft_model"]
vocab_file = config["vocab_file"]
rollback = config["rollback"]
asr_path = config["asr_path"]

"-----------------------------------------"
DELTA_ACC_THRESHOLD = 0.8
DELTA_GYRO_THRESHOLD = 1
DELTA_FLEX_THRESHOLD = 3
WINDOW_SIZE = 2   # ใช้ย้อนหลัง 5 segment
ini_state = False
classifier = RapidChangeWindowClassifier(DELTA_ACC_THRESHOLD,DELTA_GYRO_THRESHOLD,DELTA_FLEX_THRESHOLD,WINDOW_SIZE)
model = CNNTimeSeriesClassifier((50,28),51)


predictions = []
data = []
ft = []
sta = []
state = False
a = []
thes = 2
clients = []
text_list = []
isRecordingHand = ""
pyLoadResiveBuffer = ""
text = ""
hand_id = str(1595123198513) ## set by defalse
if torch.cuda.is_available():
    weight = torch.load(shl_path,weights_only=True)
    model.load_state_dict(weight)
    device = "cuda"
else:
    weight = torch.load(shl_path,weights_only=False,map_location=torch.device('cpu'))
    model.load_state_dict(weight)
    device = "cpu"
    

model.to(device)
model.double()
model.eval()

with open(rollback,"r",encoding='utf-8') as f:
        content = json.load(f)
seq = S2S(seq_name)
# f5_tts = F5TTS(
#     ckpt_file=ft_model,
#     vocab_file=vocab_file
        
# )
# tts = TTS()
asr = Thonburain(asr_path)
print("Ready to use")
"------------------------------------------"
APP_DIR = Path(__file__).resolve().parent
LOG_DIR = APP_DIR / "logs"
LOG_DIR.mkdir(parents=True, exist_ok=True)
LOG_FILE = LOG_DIR / "app.log"

logger = logging.getLogger("app_logger")
logger.setLevel(logging.INFO)
handler = RotatingFileHandler(LOG_FILE, maxBytes=2_000_000, backupCount=3, encoding="utf-8")
formatter = logging.Formatter('%(asctime)s - %(levelname)s - %(message)s')
handler.setFormatter(formatter)
logger.addHandler(handler)

app = FastAPI()

# if (APP_DIR / "static").exists():
    # app.mount("/static", StaticFiles(directory=str(APP_DIR / "static")), name="static")
app.mount("/static", StaticFiles(directory=str(APP_DIR / "static")), name="static")
templates = Jinja2Templates(directory="static")
from fastapi.middleware.cors import CORSMiddleware
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)


class PredictRequest(BaseModel):
    Id: str
    Status: int
    feature: List[List[float]]
    
class Text_voice(BaseModel):
    texts: List[str]
    metadata: Optional[dict] = None
    
class DeviceSelection(BaseModel):
    device_id: str
    timestamp: Optional[str] = None
    
    

@app.get("/", response_class=HTMLResponse)
async def index(request: Request):
    return templates.TemplateResponse("index.html", {"request": request})

@app.get("/logs", response_class=HTMLResponse)
async def view_logs(request: Request, lines: int = 200):
    n = max(1, min(10000, lines))
    if not LOG_FILE.exists():
        logs = "(log file not found yet)"
    else:
        with open(LOG_FILE, "r", encoding="utf-8", errors="ignore") as f:
            logs = "".join(f.readlines()[-n:])
    return templates.TemplateResponse("logs.html", {"request": request, "logs": logs})

def pop_from_queue_in_range(queue_obj, num_to_pop):
    """
    Removes a specified number of elements from the front of a deque-based queue.

    Args:
        queue_obj (deque): The deque object representing the queue.
        num_to_pop (int): The number of elements to remove from the front.

    Returns:
        list: A list containing the popped elements.
    """
    popped_elements = []
    for _ in range(num_to_pop):
        try:
            popped_elements.append(queue_obj.popleft())
        except IndexError:
            # Handle case where queue becomes empty before popping all requested items
            print("Queue is empty, stopped popping.")
            break
    return popped_elements


# pyLoadResiveBuffer = deque()

@app.post("/HandRecordStatus")
async def health(payload: Request):
    global isRecordingHand 
    global pyLoadResiveBuffer
    dat = await payload.json()
    if dat["status"] == True:
        isRecordingHand = True
    else:
        isRecordingHand = False
        pyLoadResiveBuffer = deque()
        
    return {"status": "samples", "info": isRecordingHand}

@app.post("/selectdevice")
async def device_id(selection: DeviceSelection):
    global hand_id
    print(selection)
    try:
        device_id = selection.device_id
        device_id = device_id.strip()
        hand_id = device_id
        
        
        if device_id == str(1595123198513):
            print("Device one true")
        
        return {"status": "samples", "info": device_id}
        
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Error processing device selection: {e}")
        raise HTTPException(status_code=500, detail=str(e))
        
    


@app.post("/predict-form")
async def predict_form(request: Request, text: str = Form(...)):
    logger.info(f"/predict called (form) with text={text}")
    result = {"received": text, "length": len(text)}
    logger.info(f"/predict-form result: {result}")
    return JSONResponse(result)


@app.websocket("/ws")
async def websocket_endpoint(websocket: WebSocket):
    await websocket.accept()
    clients.append(websocket)
    try:
        while True:
            data = await websocket.receive_text()  # not used for now
    except WebSocketDisconnect:
        clients.remove(websocket)

# API endpoint that dummy_bot can call
@app.post("/predict_hand")
async def predict_hand(payload: PredictRequest):
# async def predict_hand(payload: Request):

    # print("use predict ahnd",payload)
    
    global predictions
    global data 
    global ft
    global sta
    global state 
    global a 
    global thes 
    global text_list
    global text
    print("payload",payload.Id,"Hand_id",hand_id,isRecordingHand,len(payload.feature),len(payload.feature[0]),"len_text_list",text_list)
    print(payload.feature)
    # if isRecordingHand and payload.Id == hand_id:
    if isRecordingHand and payload.Id == hand_id:
        rows = payload.feature
        # print(rows)
        print("1")
    # logger.info(f"/aloha result: ",ft,predictions,data,sta,state)
        ft,predictions,data,sta,state = classifier.pred(rows,predictions,data,ft,state,sta)
        print("state",state,"ft",ft,len(predictions))
        
        if state==True and len(ft)>=thes and len(predictions) > 40+thes:
            
            sequence = classifier.padding(data,thes)
            # print(sequence)
            batch_x = torch.from_numpy(sequence)
            batch_x = batch_x[:,1:].unsqueeze(0)
            
            outputs = model(batch_x.double().to(device))
            text = content[str(torch.argmax(outputs).item())]
            # output = seq.predict("ศรา้งประโยค" + text) #####test
            text_list.append(text)
            predictions = []
            data = []
            ft = []
            sta = []
            state = False
            
    elif not isRecordingHand and (len(text_list) > 0):
        output = seq.predict("เรียงประโยคนี้ให้หนอย " + " ".join(text_list))
        print('2')
        print(output[0]["generated_text"])
        audio_profile = TTS(output[0]["generated_text"],
            voice="th_f_1",
            output="output.wav",
            volume=2.0,
            speed=0.8,
            
        )
        
        for i in audio_profile:
            wav = i.audio_float_array
            sr = i.sample_rate
        buf = io.BytesIO()
        buf = io.BytesIO()
        sf.write(buf, wav, sr, format="WAV")
        sf.write('output.wav', wav, sr)
        buf.seek(0)

        # Base64 encode WAV
        audio_bytes = buf.read()
        audio_b64 = base64.b64encode(audio_bytes).decode("utf-8")
        # print(audio_b64)

        for ws in clients:
            print("send to webssocke")
            await ws.send_json({"type": "bot_audio", "content": audio_b64,"text":" ".join(text_list)})
            
        text_list = []

    else:
        result = {"received":len(data)}
        return result
    return {"status": len(data)}




@app.post("/upload-audio")
async def upload_audio(request: Request):
    # file_path = os.path.join(UPLOAD_DIR, file.filename)
    wav = await request.body()
    audio = AudioSegment.from_file(io.BytesIO(wav), format="webm",)
    samples = np.array(audio.get_array_of_samples())
    samplerate = audio.frame_rate
    
    max_val = 2**(audio.sample_width * 8 - 1)
    float32_samples = samples.astype(np.float32) / max_val
    output = asr.predict(float32_samples,samplerate)

    # output,_ = librosa.load(io.BytesIO(wav))
    
    
    {"status": "samples", "info": output}
    return {"status": "samples", "info": output}


@app.post("/upload-files")
async def upload_audio(file: Annotated[UploadFile,File(description="A file read as UploadFile")]):
    # file_path = os.path.join(UPLOAD_DIR, file.filename)
    
    
    
    
    wav = await file.read()
    # print(io.BytesIO(wav))
    wav,fs = sf.read(io.BytesIO(wav))
    wav = wav.flatten()
    # output = asr.predict(wav,fs)
    time.sleep(10)
    output = "สวัสดีคับนี้คือเสียงที่เอาไว้ เทส ของทีม โมแคบ คับ"
    {"status": "samples", "info": output}
    return {"status": "samples", "info": output}



@app.get("/health")
async def health():
    return {"status": "ok"}

@app.post("/predict_csv")
async def dummy_bot(file: Annotated[UploadFile,File(description="A file read as UploadFile")]):


    global predictions
    global data 
    global ft
    global sta
    global state 
    global a 
    global thes 
    global text_list
    global text
    
    contents = await file.read()
    csv_string = contents.decode('utf-8')
    csv_file = io.StringIO(csv_string)

    df = pd.read_csv(csv_file)
    
    for i in range(0,len(df),25):
        rows = df.iloc[i*25:(i+1)*25].values[:,:29 - len(df.columns)]
    # logger.info(f"/aloha result: ",ft,predictions,data,sta,state)
        ft,predictions,data,sta,state = classifier.pred(rows,predictions,data,ft,state,sta)
        # logger.info(f"/aloha result-end: ",ft,predictions,data,sta,state)
        
        
        if state==True and len(ft)>=thes and len(predictions) > 40+thes:
            
            sequence = classifier.padding(data,thes)
            # print(sequence)
            batch_x = torch.from_numpy(sequence)
            batch_x = batch_x[:,1:].unsqueeze(0)
            
            outputs = model(batch_x.double().to(device))
            text = content[str(torch.argmax(outputs).item())]
        
        
        
            # output = seq.predict("ศรา้งประโยค" + text) #####test
            text_list.append(text)
            predictions = []
            data = []
            ft = []
            sta = []
            state = False
    output = seq.predict("เรียงประโยคนี้ให้หนอย " + " ".join(text_list))
    
    demo_text = "ฉันกินอาหาร"
    # audio_profile = TTS(output[0]["generated_text"],
    audio_profile = TTS(demo_text,
        voice="th_f_1",
        output="output.wav",
        volume=2.0,
        speed=0.8,
        
    )
    
    for i in audio_profile:
        wav = i.audio_float_array
        sr = i.sample_rate
    buf = io.BytesIO()
    buf = io.BytesIO()
    sf.write(buf, wav, sr, format="WAV")
    sf.write('output.wav', wav, sr)
    buf.seek(0)

    # Base64 encode WAV
    audio_bytes = buf.read()
    audio_b64 = base64.b64encode(audio_bytes).decode("utf-8")
    # print(audio_b64)
    
    for ws in clients:
        await ws.send_json({"type": "bot_audio", "content": audio_b64,"text":demo_text})

            

    else:
        result = {"received":len(data)}
        return result
