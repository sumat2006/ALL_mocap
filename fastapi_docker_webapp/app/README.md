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

**Varibale describe**
```python
predictions = [] 
data = []
ft = []
sta = []
state = False
a = []
thes = 2
clients = []
text_list = []

---------------------------------------------------------------------------

def pred(self,rows,predictions,data,ft,state,sta):
    for row in rows:
        # print(row)
        st = self.classify(row)
        
        a.append(st)
        if st :### if predict true 
            state = True
            ### add data re false array and add true array
            predictions.append(st)
            data.append(row)
            ft = []
            sta.append(st)
        elif not st and state: ### if predict false and previous true
            ### add data and add false array
            
            ft.append(st)
            predictions.append(st)
            data.append(row)
        
    return ft,predictions,data,sta,state

```
## pred function explain
> row is a data that receive from user

> st is a state of current input. High or Low

> state use to indicate that glove in really activate

> data is a data for output data

> ft is false state to show that how many time false appear continuously

> sta is a dummy variable

> prediction is a list or state for every rows



```python
def padding(self,data,thes):
        answer = []
        features = np.array(data[:-thes])
        if len(features) >= 50:
            # If longer than chunk_size, use uniform sampling
            indices = np.linspace(0, len(features)-1, 50, dtype=int)
            sequence = features[indices]
        else:
            # If shorter, pad with zeros at the end
            sequence = np.zeros((50, features.shape[1]))
            sequence[:len(features)] = features
        return sequence
```

## padding function
> padding is a fucntion that will extend data if it amout of classified data not reaches the threshold, and cut if classified data is exceed the threshold.



----------------------------------------------------------------------------------------

# Java script explaine

🎤 Audio Recording & Handling

**initializeAudio()**
Sets up microphone access, initializes MediaRecorder, handles data chunks, and processes recordings on stop.

**startRecording()**
Begins audio recording with UI updates, timer, and MediaRecorder.start().

***stopRecording()***
Stops recording, resets UI, clears timer, and calls MediaRecorder.stop().

**startHandRecording()**
Starts a “hand recording” (external device status recording), verifies correct device, updates UI, and starts timer.

**stopHandRecording()**
Stops hand recording, updates device status, resets UI and timer.

**handleAudioRecording(audioBlob)**
Adds user’s audio message to chat, shows a spinner, uploads the audio to /upload-audio, then replaces spinner with server response.

**updateRecordingTimer()**
Updates recording duration display (MM:SS).

🔊 Audio Playback

**createAudioPlayer(audioBlob, messageId)**
Generates an audio player UI with waveform, play/pause buttons, and progress bar.

**toggleAudioPlayback(audioUrl, messageId**)
Handles play/pause of an audio file, manages multiple audio instances, updates UI, and handles errors.

**stopCurrentAudio()**
Stops currently playing audio and resets UI elements (progress bar, waveform, etc.).

**seekAudio(event, messageId)**
Allows clicking on the progress bar to jump to a specific timestamp.

**formatTime(seconds)**
Formats seconds into MM:SS.

💬 Chat Messaging

**addMessage(text, isUser = false, isAudio = false, audioBlob = null)**
Adds a new chat message (text or audio) to the conversation with avatar and timestamp.

**sendMessage()**
Sends a text message, shows a bot spinner, then simulates a random bot response after 2s.

**callDummyBot()**
Calls /dummy_bot, retrieves its response, and adds it as a bot message.

⏳ Spinner System

**addBotSpinner(message, spinnerType)**
Inserts a spinner message (loading dots or spinning circle) into the chat and tracks it in activeSpinners.

**removeBotSpinner(spinnerId, finalMessage, type)**
Removes a spinner and replaces it with the final bot or user message.

**updateBotSpinner(spinnerId, newMessage)**
Updates text inside an existing spinner.

**removeAllSpinners()**
Clears all active spinners from chat.

📂 File Upload & Display

**createFileMessage(file, messageId)**
Generates a UI preview for an uploaded file (CSV or MP3) with metadata and a download button.

**addFileMessage(files, isUser = true)**
Adds one or multiple uploaded files as chat messages.

📑 UI & Sidebar

**toggleSidebar()**
Handles mobile vs. desktop sidebar toggle (slides in/out, expands/collapses main content).

✅ Summarize:

Audio recording & playback (initialize, start/stop recording, play/pause, waveform, seek).

Chat messaging (user messages, bot responses, dummy bot).

Spinner system (loading indicators for bot processing).

File upload system (preview and download).

Sidebar UI management.