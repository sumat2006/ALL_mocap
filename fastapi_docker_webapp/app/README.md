# Main Programs

```
/predict-hand
use in api calling to do a hand prediction from row-by-row data.
```


```
/predict_csv
use in api calling to do a hand prediction using csv data.
```


```
/upload-audio
Use in ASR that receive audio input as a webm file from web application.
```

```
/upload-files
Receive file uploaded and then do trasncripe.
```


```
/HandRecordStatus
gives a current status of glove, output is either true or false.
```



Websocket is use to send an audioblob to the web apllication in case of output is audio, e.g. speech from hand sign prediction.

we use fastapi file upload to receive uploaded file from wep application. the fast api will call a static folder to load files in static e.g. index.html, javascript.html

The icons in the web application are hand-sign, speech , upload file, send text



# Web application text description

**RapidChangeClassifier:** 
> is used to classify if current value is active or not-active using sliding window, sliding window use the last N input to classify if current is active or not.
This function requires Acc,Gyro,Flex threshold and window size that denote by N.

**CNNTimeserieClassifier:**
> is a main model that we use in this program to classify sign language, it requires  to put ((chunk_size,feature),output) to clarify the model.
standard configuration is ((50,28),51)

**S2S:**
> function use to initial Language model for text ordering.

in file main.py line 57 thes is used to delete the false input from data i.e. classifier need to receive N false value to stop.

PredictRequest(BaseModel) and Text_voice(BaseModel) indicate the json column values



