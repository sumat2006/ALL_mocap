// Global variables
let sidebarCollapsed = false;
let mediaRecorder = null;
let audioChunks = [];
let isRecordingHand = false;    
let isRecording = false;
let recordingTimer = null;
let recordingStartTime = null;
let currentAudio = null;
let currentAudioId = null;
let audioInstances = new Map();

let userPerMissionId = 1595123198513;
let deviceStatus = false
const devices = [
  { id: 1595123198513, label: 'MOCAP_test_device_1' },
  { id: 5465198512316, label: 'MOCAP_test_device_2' },
  { id: 1951195195198, label: 'MOCAP_test_device_3' },
  { id: 1695198515615, label: 'MOCAP_test_device_4' }
];

// Global variables for spinner system
let activeSpinners = new Map();

// Global variables for progress tracking
let activeProgressMessages = new Map();
let progressUpdateIntervals = new Map();

// Store uploaded files for download
const uploadedFiles = new Map();

// CSS for the spinner system
const spinnerCSS = `
.bot-spinner {
    display: inline-flex;
    align-items: center;
    gap: 8px;
}

.spinner {
    width: 16px;
    height: 16px;
    border: 2px solid #e2e8f0;
    border-top: 2px solid #3b82f6;
    border-radius: 50%;
    animation: spin 1s linear infinite;
}

@keyframes spin {
    from { transform: rotate(0deg); }
    to { transform: rotate(360deg); }
}

.spinner-text {
    color: #64748b;
    font-style: italic;
}

.typing-dots {
    display: inline-flex;
    gap: 2px;
}

.typing-dots span {
    width: 4px;
    height: 4px;
    background: #64748b;
    border-radius: 50%;
    animation: typingDots 1.4s infinite ease-in-out;
}

.typing-dots span:nth-child(1) { animation-delay: -0.32s; }
.typing-dots span:nth-child(2) { animation-delay: -0.16s; }
.typing-dots span:nth-child(3) { animation-delay: 0s; }

@keyframes typingDots {
    0%, 80%, 100% {
        transform: scale(0.8);
        opacity: 0.5;
    }
    40% {
        transform: scale(1);
        opacity: 1;
    }
}
`;

// Inject CSS if not already added
if (!document.getElementById('spinner-styles')) {
    const styleSheet = document.createElement('style');
    styleSheet.id = 'spinner-styles';
    styleSheet.textContent = spinnerCSS;
    document.head.appendChild(styleSheet);
}

// Get the select element from the DOM
const selectElement = document.getElementById('deviceSelect');
// Loop through the data and create an option for each item
devices.forEach(device => {
    // Create a new option element
    const option = document.createElement('option');

    // Set the value and the text of the option
    option.value = device.id;
    option.textContent = device.label;

    // Add the new option to the select bar
    selectElement.appendChild(option);
});

async function sendToBackend(deviceId) {
    console.log(deviceId)
    try {
        // Replace with your actual API endpoint
        const response = await fetch('/selectdevice', {
            method: 'POST',
            headers: {
                'Content-Type': 'application/json',
            },
            body: JSON.stringify({
                device_id: deviceId,
            })
        });

        if (response.ok) {
            const result = await response.json();
            console.log('Backend response:', result);
        } else {
            console.error('Failed to send to backend');
        }
    } catch (error) {
        console.error('Error sending to backend:', error);
        // Still works locally even if backend is not available
    }
}


selectElement.addEventListener('change', function() {
    // Get the selected device ID
    selectedDeviceId = this.value;
    console.log(selectedDeviceId);
    sendToBackend(selectedDeviceId);

    if (selectedDeviceId == 1595123198513)
        {deviceStatus = true;}
    else {deviceStatus = false;}

    console.log(deviceStatus)
    if(deviceStatus){
        statusDot.className = "status-dot online"
        document.getElementById("status_text").innerHTML  = "Connected"
    }
    else{
        statusDot.className = "status-dot offlin"
        document.getElementById("status_text").innerHTML  = "Disconnect"
    }

});




// DOM elements
const statusDot = document.getElementById("statusDot")
const messageInput = document.getElementById('messageInput');
const sendButton = document.getElementById('sendButton');
const recordButton = document.getElementById('recordButton');
const recordHandButton = document.getElementById('recordHandButton');
const chatMessages = document.getElementById('chatMessages');
const recordingIndicator = document.getElementById('recordingIndicator');
const recordingTimerDisplay = document.getElementById('recordingTimer');
const sidebar = document.getElementById('sidebar');
const mainContent = document.getElementById('mainContent');
const sidebarOverlay = document.getElementById('sidebarOverlay');
const fileInput = document.getElementById('fileInput');
const fileInfo = document.getElementById('fileInfo');
const fileList = document.getElementById('fileList');



// Initialize audio recording
async function initializeAudio() {
    try {
        const stream = await navigator.mediaDevices.getUserMedia({ audio: true });
        mediaRecorder = new MediaRecorder(stream);
        
        mediaRecorder.ondataavailable = (event) => {
            if (event.data.size > 0) {
                audioChunks.push(event.data);
                console.log(event.data);
            }
        };
        
        mediaRecorder.onstop = () => {
            const audioBlob = new Blob(audioChunks, { type: 'audio/wav' });
            audioChunks = [];
            handleAudioRecording(audioBlob);
        };
        
    } catch (error) {
        console.error('Error accessing microphone:', error);
        alert('Microphone access is required for voice recording. Please allow microphone access and refresh the page.');
    }
}

// Enhanced toggle sidebar function
function toggleSidebar() {
    const isMobile = window.innerWidth <= 768;
    
    if (isMobile) {
        // Mobile behavior - toggle from top
        const isActive = sidebar.classList.contains('active');
        
        if (isActive) {
            // Hide sidebar
            sidebar.classList.remove('active');
            sidebarOverlay.classList.remove('active');
            document.body.style.overflow = '';
        } else {
            // Show sidebar
            sidebar.classList.add('active');
            sidebarOverlay.classList.add('active');
            document.body.style.overflow = 'hidden'; // Prevent background scrolling
        }
    } else {
        // Desktop behavior - toggle from left
        sidebarCollapsed = !sidebarCollapsed;
        
        if (sidebarCollapsed) {
            sidebar.classList.add('collapsed');
            mainContent.classList.add('expanded');
        } else {
            sidebar.classList.remove('collapsed');
            mainContent.classList.remove('expanded');
        }
    }
}

// Create audio player component
function createAudioPlayer(audioBlob, messageId) {
    const audioUrl = URL.createObjectURL(audioBlob);
    
    // Generate random waveform bars for visual effect
    const waveformBars = Array.from({length: 20}, () => Math.random() * 15 + 5)
        .map(height => `<div class="waveform-bar" style="height: ${height}px;"></div>`)
        .join('');
    
    return `
        <div class="audio-message">
            <div class="audio-player">
                <button class="audio-play-btn" data-audio-id="${messageId}" onclick="toggleAudioPlayback('${audioUrl}', '${messageId}')">
                    <!-- Play Icon -->
                    <svg class="play-icon" viewBox="0 0 20 20" fill="currentColor">
                        <path fill-rule="evenodd" d="M10 18a8 8 0 100-16 8 8 0 000 16zM9.555 7.168A1 1 0 008 8v4a1 1 0 001.555.832l3-2a1 1 0 000-1.664l-3-2z"/>
                    </svg>
                    <!-- Pause Icon -->
                    <svg class="pause-icon" viewBox="0 0 20 20" fill="currentColor">
                        <path fill-rule="evenodd" d="M18 10a8 8 0 11-16 0 8 8 0 0116 0zM7 8a1 1 0 012 0v4a1 1 0 11-2 0V8zm5-1a1 1 0 00-1 1v4a1 1 0 102 0V8a1 1 0 00-1-1z"/>
                    </svg>
                    <!-- Loading Icon -->
                    <svg class="loading-icon" viewBox="0 0 20 20" fill="currentColor">
                        <path fill-rule="evenodd" d="M4 2a1 1 0 011 1v2.101a7.002 7.002 0 0111.601 2.566 1 1 0 11-1.885.666A5.002 5.002 0 005.999 7H9a1 1 0 010 2H4a1 1 0 01-1-1V3a1 1 0 011-1zm.008 9.057a1 1 0 011.276.61A5.002 5.002 0 0014.001 13H11a1 1 0 110-2h5a1 1 0 011 1v5a1 1 0 11-2 0v-2.101a7.002 7.002 0 01-11.601-2.566 1 1 0 01.61-1.276z"/>
                    </svg>
                </button>
                
                <div class="audio-info">
                    <div class="audio-progress-container">
                        <div class="audio-progress" onclick="seekAudio(event, '${messageId}')">
                            <div class="audio-progress-bar" id="progress-${messageId}"></div>
                        </div>
                        <div class="audio-time" id="time-${messageId}">0:00</div>
                    </div>
                    <div class="audio-waveform" id="waveform-${messageId}">
                        ${waveformBars}
                    </div>
                </div>
            </div>
        </div>
    `;
}

// Enhanced audio playback with proper play/pause functionality
function toggleAudioPlayback(audioUrl, messageId) {
    const button = document.querySelector(`[data-audio-id="${messageId}"]`);
    const progressBar = document.getElementById(`progress-${messageId}`);
    const timeDisplay = document.getElementById(`time-${messageId}`);
    const waveformBars = document.querySelectorAll(`#waveform-${messageId} .waveform-bar`);
    
    // If another audio is playing, stop it first
    if (currentAudio && currentAudioId !== messageId && !currentAudio.paused) {
        stopCurrentAudio();
    }
    
    // Check if this audio exists and is currently playing
    if (audioInstances.has(messageId)) {
        const audio = audioInstances.get(messageId);
        if (!audio.paused) {
            // Audio is playing, so pause it
            audio.pause();
            button.classList.remove('playing');
            currentAudio = null;
            currentAudioId = null;
            return;
        } else {
            // Audio is paused, so resume it
            audio.play().then(() => {
                button.classList.add('playing');
                currentAudio = audio;
                currentAudioId = messageId;
            }).catch(error => {
                console.error('Error resuming audio:', error);
                button.classList.remove('playing');
            });
            return;
        }
    }
    
    // Create new audio instance if doesn't exist
    const audio = new Audio(audioUrl);
    audioInstances.set(messageId, audio);
    
    // Set up event listeners for this audio instance
    audio.addEventListener('loadstart', () => {
        button.classList.add('loading');
    });
    
    audio.addEventListener('canplay', () => {
        button.classList.remove('loading');
    });
    
    audio.addEventListener('timeupdate', () => {
        if (currentAudioId === messageId) {
            const progress = (audio.currentTime / audio.duration) * 100;
            progressBar.style.width = `${progress || 0}%`;
            
            // Update time display
            const currentTime = formatTime(audio.currentTime);
            timeDisplay.textContent = currentTime;
            
            // Animate waveform bars
            const activeIndex = Math.floor((audio.currentTime / audio.duration) * waveformBars.length);
            waveformBars.forEach((bar, index) => {
                if (index <= activeIndex) {
                    bar.classList.add('active');
                } else {
                    bar.classList.remove('active');
                }
            });
        }
    });
    
    audio.addEventListener('ended', () => {
        button.classList.remove('playing');
        progressBar.style.width = '0%';
        timeDisplay.textContent = '0:00';
        waveformBars.forEach(bar => bar.classList.remove('active'));
        currentAudio = null;
        currentAudioId = null;
        // Reset audio to beginning
        audio.currentTime = 0;
    });
    
    audio.addEventListener('error', (e) => {
        console.error('Audio playback error:', e);
        button.classList.remove('loading', 'playing');
        currentAudio = null;
        currentAudioId = null;
        alert('Error playing audio. Please try again.');
    });
    
    // Start playing the audio
    button.classList.add('loading');
    audio.play().then(() => {
        button.classList.remove('loading');
        button.classList.add('playing');
        currentAudio = audio;
        currentAudioId = messageId;
    }).catch(error => {
        console.error('Error playing audio:', error);
        button.classList.remove('loading', 'playing');
        alert('Error playing audio. Please try again.');
    });
}

// Stop currently playing audio
function stopCurrentAudio() {
    if (currentAudio) {
        currentAudio.pause();
        // Find and update the button state
        if (currentAudioId) {
            const button = document.querySelector(`[data-audio-id="${currentAudioId}"]`);
            if (button) {
                button.classList.remove('playing');
            }
            
            // Reset progress bar and waveform
            const progressBar = document.getElementById(`progress-${currentAudioId}`);
            const timeDisplay = document.getElementById(`time-${currentAudioId}`);
            const waveformBars = document.querySelectorAll(`#waveform-${currentAudioId} .waveform-bar`);
            
            if (progressBar) progressBar.style.width = '0%';
            if (timeDisplay) timeDisplay.textContent = '0:00';
            if (waveformBars) waveformBars.forEach(bar => bar.classList.remove('active'));
        }
        
        currentAudio = null;
        currentAudioId = null;
    }
}

// Seek audio to specific position
function seekAudio(event, messageId) {
    const audio = audioInstances.get(messageId);
    if (!audio) return;
    
    const progressBar = event.currentTarget;
    const rect = progressBar.getBoundingClientRect();
    const percent = (event.clientX - rect.left) / rect.width;
    const newTime = percent * audio.duration;
    
    if (isFinite(newTime)) {
        audio.currentTime = newTime;
    }
}

// Format time for display
function formatTime(seconds) {
    if (!isFinite(seconds)) return '0:00';
    const mins = Math.floor(seconds / 60);
    const secs = Math.floor(seconds % 60);
    return `${mins}:${secs.toString().padStart(2, '0')}`;
}

// Add message to chat
function addMessage(text, isUser = false, isAudio = false, audioBlob = null) {
    const messageDiv = document.createElement('div');
    const messageId = 'msg-' + Date.now() + '-' + Math.random().toString(36).substr(2, 9);
    messageDiv.className = `message ${isUser ? 'user-message' : 'bot-message'}`;
    
    const currentTime = new Date().toLocaleTimeString([], {hour: '2-digit', minute:'2-digit'});
    
    let messageContent = '';
    if (isAudio && audioBlob) {
        messageContent = createAudioPlayer(audioBlob, messageId);
    } else {
        messageContent = `<div class="message-text">${text}</div>`;
    }
    
    messageDiv.innerHTML = `
        <div class="message-avatar">
            <div class="avatar-icon">${isUser ? '👤' : '🤖'}</div>
        </div>
        <div class="message-content">
            ${messageContent}
            <div class="message-time">${currentTime}</div>
        </div>
    `;
    
    chatMessages.appendChild(messageDiv);
    chatMessages.scrollTop = chatMessages.scrollHeight;
}

// SPINNER SYSTEM FUNCTIONS

// Add bot response with spinner
function addBotSpinner(message = "Processing your request", spinnerType = "spinner") {
    const messageDiv = document.createElement('div');
    const messageId = 'spinner-' + Date.now() + '-' + Math.random().toString(36).substr(2, 9);
    messageDiv.className = 'message bot-message';
    messageDiv.id = messageId;
    
    const currentTime = new Date().toLocaleTimeString([], {hour: '2-digit', minute:'2-digit'});
    
    let spinnerHTML = '';
    if (spinnerType === "spinner") {
        spinnerHTML = `
            <div class="bot-spinner">
                <div class="spinner"></div>
                <span class="spinner-text">${message}</span>
            </div>
        `;
    } else if (spinnerType === "dots") {
        spinnerHTML = `
            <div class="bot-spinner">
                <span class="spinner-text">${message}</span>
                <div class="typing-dots">
                    <span></span>
                    <span></span>
                    <span></span>
                </div>
            </div>
        `;
    }
    
    messageDiv.innerHTML = `
        <div class="message-avatar">
            <div class="avatar-icon">🤖</div>
        </div>
        <div class="message-content">
            <div class="message-text">
                ${spinnerHTML}
            </div>
            <div class="message-time">${currentTime}</div>
        </div>
    `;
    
    chatMessages.appendChild(messageDiv);
    chatMessages.scrollTop = chatMessages.scrollHeight;
    
    // Store reference for later removal
    activeSpinners.set(messageId, {
        element: messageDiv,
        message: message,
        startTime: Date.now()
    });
    
    return messageId;
}

// Remove spinner and add actual bot response
function removeBotSpinner(spinnerId, finalMessage,type="bot") {
    const spinnerInfo = activeSpinners.get(spinnerId);
    if (!spinnerInfo) return;
    
    // Remove the spinner message
    if (spinnerInfo.element && spinnerInfo.element.parentNode) {
        spinnerInfo.element.parentNode.removeChild(spinnerInfo.element);
    }
    
    // Add the actual bot response
    if (finalMessage !=""){
        if (type == "bot"){
        addMessage(finalMessage, false);
        }else{
        addMessage(finalMessage, true);
        }
    }
    
    // Clean up
    activeSpinners.delete(spinnerId);
}

// Update spinner message
function updateBotSpinner(spinnerId, newMessage) {
    const spinnerInfo = activeSpinners.get(spinnerId);
    if (!spinnerInfo) return;
    
    const spinnerText = spinnerInfo.element.querySelector('.spinner-text');
    if (spinnerText) {
        spinnerText.textContent = newMessage;
        spinnerInfo.message = newMessage;
    }
}

// Remove all active spinners
function removeAllSpinners() {
    activeSpinners.forEach((spinnerInfo, spinnerId) => {
        if (spinnerInfo.element && spinnerInfo.element.parentNode) {
            spinnerInfo.element.parentNode.removeChild(spinnerInfo.element);
        }
    });
    activeSpinners.clear();
}

// Modified handleAudioRecording with spinner
async function handleAudioRecording(audioBlob) {
    try {
        // Add audio message to chat first
        addMessage('', true, true, audioBlob);
        
        // Show spinner while processing
        const spinnerId = addBotSpinner("Processing your audio message", "spinner");
        
        const arrayBuffer = await audioBlob.arrayBuffer();
        const response = await fetch("/upload-audio", {
            method: "POST",
            headers: {
                "Content-Type": "audio/wav"
            },
            body: arrayBuffer
        });

        const result = await response.json();
        console.log("Server response:", result);

        // Remove spinner and show result
        removeBotSpinner(spinnerId, result.info,"user");

    } catch (err) {
        // Remove spinner and show error
        const activeSpinnerIds = Array.from(activeSpinners.keys());
        if (activeSpinnerIds.length > 0) {
            removeBotSpinner(activeSpinnerIds[activeSpinnerIds.length - 1], "There was an error processing your audio. Please try again.","user");
        }
        console.error("Upload failed:", err);
    }
}

async function callDummyBot() {
    try {
        const response = await fetch("/dummy_bot");
        const data = await response.json();
        // Call addMessage as a bot (not user)
        addMessage(data.message, false);
    } catch (err) {
        console.error("API call failed", err);
    }
}

// Start recording
function startRecording() {
    if (!mediaRecorder) {
        alert('Microphone not initialized. Please refresh the page and allow microphone access.');
        return;
    }
    
    // Stop any currently playing audio
    stopCurrentAudio();
    
    isRecording = true;
    recordButton.className = 'record-button recording';
    recordingIndicator.classList.add('active');
    recordingStartTime = Date.now();
    
    // Start recording timer
    recordingTimer = setInterval(updateRecordingTimer, 100);
    
    mediaRecorder.start();
}

// Stop recording
function stopRecording() {
    if (!isRecording) return;
    
    isRecording = false;
    recordButton.className = 'record-button processing';
    recordingIndicator.classList.remove('active');
    
    clearInterval(recordingTimer);
    recordingTimerDisplay.textContent = '00:00';
    
    mediaRecorder.stop();
    
    // Reset button after processing
    setTimeout(() => {
        recordButton.className = 'record-button idle';
    }, 1000);
}

async function startHandRecording() {
    console.log(getSelectedDevice())
    console.log(userPerMissionId)

    const response = await fetch("/HandRecordStatus",{
        method: "POST",
            body:JSON.stringify({ status: true })
        }
    )
    const data = await response.json();
    console.log(data)

    if (getSelectedDevice() != userPerMissionId){
        showError("Please select the correct device to start recording.")
        return;
    }

    // Stop any currently playing audio
    stopCurrentAudio();

    isRecordingHand = true;
    recordHandButton.className = 'record-hand-button recording';
    recordingIndicator.classList.add('active');
    recordingStartTime = Date.now();

    // Start recording timer
    recordingTimer = setInterval(updateRecordingTimer, 100);
}

async function stopHandRecording() {
    isRecordingHand = false;
    const response = await fetch("/HandRecordStatus",{
        method: "POST",
        body:JSON.stringify({ status: false }),
        }
    )
    const data = await response.json();
    console.log(data)

    recordHandButton.className = 'record-hand-button idle';
    recordingIndicator.classList.remove('active');

    clearInterval(recordingTimer);
    recordingTimerDisplay.textContent = '00:00';

    // Reset button after processing
    setTimeout(() => {
        recordHandButton.className = 'record-hand-button idle';
    }, 1000);
}

// Update recording timer
function updateRecordingTimer() {
    if (!recordingStartTime) return;
    
    const elapsed = Math.floor((Date.now() - recordingStartTime) / 1000);
    const minutes = Math.floor(elapsed / 60).toString().padStart(2, '0');
    const seconds = (elapsed % 60).toString().padStart(2, '0');
    recordingTimerDisplay.textContent = `${minutes}:${seconds}`;
}

// Modified send message with spinner for bot responses
function sendMessage() {
    const text = messageInput.value.trim();
    if (!text) return;
    
    // Stop any currently playing audio
    stopCurrentAudio();
    
    // Add user message
    addMessage(text, true);
    messageInput.value = '';
    
    // Show spinner while "thinking"
    const spinnerId = addBotSpinner("Thinking", "dots");
    
    // Simulate bot processing time
    setTimeout(() => {
        const responses = [
            "I received your message. How can I assist you further?",
            "That's interesting! Tell me more about what you need.",
            "I'm processing your request. Please wait a moment...",
            "Thanks for your input! Here's what I found:",
            "I understand. Let me help you with that."
        ];
        const randomResponse = responses[Math.floor(Math.random() * responses.length)];
        
        // Remove spinner and show response
        removeBotSpinner(spinnerId, randomResponse,"bot");
    }, 2000);
}

// Create file message displays in chat
function createFileMessage(file, messageId) {
    const fileType = file.name.toLowerCase().endsWith('.csv') ? 'CSV' : 'MP3';
    const fileSize = (file.size / 1024 / 1024).toFixed(2);
    const fileIcon = fileType === 'CSV' ? '📊' : '🎵';
    
    return `
        <div class="file-message">
            <div class="file-preview">
                <div class="file-icon">${fileIcon}</div>
                <div class="file-details">
                    <div class="file-name">${file.name}</div>
                    <div class="file-meta">
                        <span class="file-type">${fileType}</span> • 
                        <span class="file-size">${fileSize} MB</span>
                    </div>
                </div>
                <div class="file-actions">
                    <button class="file-action-btn" onclick="downloadFile('${messageId}', '${file.name}')">
                        <svg width="16" height="16" viewBox="0 0 16 16" fill="currentColor">
                            <path d="M8 12l-4-4h3V4h2v4h3l-4 4z"/>
                            <path d="M2 14h12v2H2z"/>
                        </svg>
                    </button>
                </div>
            </div>
        </div>
    `;
}

// Add file messages to chat
function addFileMessage(files, isUser = true) {
    Array.from(files).forEach(file => {
        const messageDiv = document.createElement('div');
        const messageId = 'file-' + Date.now() + '-' + Math.random().toString(36).substr(2, 9);
        messageDiv.className = `message ${isUser ? 'user-message' : 'bot-message'}`;
        
        // Store file for potential download
        uploadedFiles.set(messageId, file);
        
        const currentTime = new Date().toLocaleTimeString([], {hour: '2-digit', minute:'2-digit'});
        
        messageDiv.innerHTML = `
            <div class="message-avatar">
                <div class="avatar-icon">${isUser ? '👤' : '🤖'}</div>
            </div>
            <div class="message-content">
                ${createFileMessage(file, messageId)}
                <div class="message-time">${currentTime}</div>
            </div>
        `;
        
        chatMessages.appendChild(messageDiv);
        chatMessages.scrollTop = chatMessages.scrollHeight;
    });
}

// Download function for files
function downloadFile(messageId, fileName) {
    const file = uploadedFiles.get(messageId);
    if (file) {
        const url = URL.createObjectURL(file);
        const a = document.createElement('a');
        a.href = url;
        a.download = fileName;
        document.body.appendChild(a);
        a.click();
        document.body.removeChild(a);
        URL.revokeObjectURL(url);
    }
}

function hideFileInfo() {
    fileInfo.classList.remove('show');
    setTimeout(() => {
        fileInfo.style.display = 'none';
    }, 300);
}

// Handle file processing
function handleFileUpload(files) {
    Array.from(files).forEach(file => {
        if (file.type === 'text/csv' || file.name.toLowerCase().endsWith('.csv')) {
            console.log('Processing CSV file:', file.name);
            processCsvFile(file);
        } else if (file.type === 'audio/mp3' || file.name.toLowerCase().endsWith('.mp3')) {
            console.log('Processing MP3 file:', file.name);
            processMp3File(file);
        }
    });
}

function processCsvFile(file) {
    const reader = new FileReader();
    reader.onload = function(e) {
        const csvContent = e.target.result;
        console.log('CSV content loaded:', csvContent.substring(0, 100) + '...');
    };
    reader.readAsText(file);
}

function processMp3File(file) {
    const reader = new FileReader();
    reader.onload = function(e) {
        const audioData = e.target.result;
        console.log('MP3 file loaded, size:', audioData.byteLength);
    };
    reader.readAsArrayBuffer(file);
}

// Modified file upload with spinner
async function handleFileUploadWithSpinner(files) {
    const fileCount = files.length;
    const name = files[0].name.toLowerCase()
    if (name.endsWith(".mp3")) fileType = "MP3";
    else if (name.endsWith(".wav")) fileType = "WAV";
    else if (name.endsWith(".ogg")) fileType = "OGG";
    else if (name.endsWith(".flac")) fileType = "FLAC";
    else if (name.endsWith(".csv")) fileType = "csv";
    
    
    // Show spinner
    const spinnerId = addBotSpinner(`Processing ${fileCount} file${fileCount > 1 ? 's' : ''}`, "dots");

    try {
        const formData = new FormData();
        formData.append("file", files[0]);
        // Update spinner message
        updateBotSpinner(spinnerId, "Uploading files to server");
        


        if (fileType != "csv") {
        const response = await fetch("/upload-files", {
            method: "POST",
            body: formData
        });

        if (response.ok) {
            const result = await response.json();
            removeBotSpinner(spinnerId, result.info,"user");
        } else {
            throw new Error(`Upload failed: ${response.statusText}`);
        }
        }else{
            
            const response = await fetch("/predict_csv", {
                method: "POST",
                body: formData
            });
            removeBotSpinner(spinnerId, "","user");
        }
    } catch (error) {
        console.error("File upload error:", error);
        removeBotSpinner(spinnerId, "Sorry, there was an error processing your files. Please try again.","bot");
    }
}

// Event listeners
document.addEventListener('DOMContentLoaded', () => {
    // Close sidebar when overlay is clicked (mobile)
    sidebarOverlay.addEventListener('click', () => {
        if (window.innerWidth <= 768) {
            sidebar.classList.remove('active');
            sidebarOverlay.classList.remove('active');
            document.body.style.overflow = '';
        }
    });

    // Close sidebar when clicking outside on mobile
    document.addEventListener('click', (e) => {
        if (window.innerWidth <= 768) {
            const isClickInsideSidebar = sidebar.contains(e.target);
            const isToggleButton = e.target.closest('.toggle-btn');
            const isSidebarActive = sidebar.classList.contains('active');
            
            if (!isClickInsideSidebar && !isToggleButton && isSidebarActive) {
                sidebar.classList.remove('active');
                sidebarOverlay.classList.remove('active');
                document.body.style.overflow = '';
            }
        }
    });

    // Handle escape key to close sidebar on mobile
    document.addEventListener('keydown', (e) => {
        if (e.key === 'Escape' && window.innerWidth <= 768 && sidebar.classList.contains('active')) {
            sidebar.classList.remove('active');
            sidebarOverlay.classList.remove('active');
            document.body.style.overflow = '';
        }
    });

    recordHandButton.addEventListener('click', () => {
        if (isRecordingHand) {
            stopHandRecording();
        } else {
            startHandRecording();
        }
    });

    recordButton.addEventListener('click', () => {
        if (isRecording) {
            stopRecording();
        } else {
            startRecording();
        }
    });

    sendButton.addEventListener('click', sendMessage);

    messageInput.addEventListener('keypress', (e) => {
        if (e.key === 'Enter') {
            sendMessage();
        }
    });

    // Navigation click handlers
    document.querySelectorAll('.nav-link').forEach(link => {
        link.addEventListener('click', function(e) {
            e.preventDefault();
            document.querySelectorAll('.nav-link').forEach(l => l.classList.remove('active'));
            this.classList.add('active');
            console.log('Navigated to:', this.textContent.trim());
            
            // Close sidebar on mobile after navigation
            if (window.innerWidth <= 768) {
                sidebar.classList.remove('active');
                sidebarOverlay.classList.remove('active');
                document.body.style.overflow = '';
            }
        });
    });

    // Mobile responsiveness
    function handleResize() {
        if (window.innerWidth <= 768) {
            // Mobile mode
            sidebar.classList.remove('collapsed');
            sidebar.classList.remove('active');
            mainContent.classList.add('expanded');
            sidebarOverlay.classList.remove('active');
            document.body.style.overflow = '';
        } else {
            // Desktop mode
            sidebar.classList.remove('active');
            sidebarOverlay.classList.remove('active');
            document.body.style.overflow = '';
            
            if (!sidebarCollapsed) {
                sidebar.classList.remove('collapsed');
                mainContent.classList.remove('expanded');
            } else {
                sidebar.classList.add('collapsed');
                mainContent.classList.add('expanded');
            }
        }
    }

    window.addEventListener('resize', handleResize);

    // Initialize on page load
    handleResize();
    initializeAudio();

    // Prevent body scroll when sidebar is open on mobile
    function preventBodyScroll(e) {
        if (sidebar.classList.contains('active') && window.innerWidth <= 768) {
            if (!sidebar.contains(e.target)) {
                e.preventDefault();
            }
        }
    }

    document.addEventListener('touchmove', preventBodyScroll, { passive: false });

    // WebSocket connection
    let wsProtocol = window.location.protocol === "https:" ? "wss:" : "ws:";
    let wsHost = window.location.host;

    const ws = new WebSocket(`${wsProtocol}//${wsHost}/ws`);
    ws.onmessage = function(event) {
        const data = JSON.parse(event.data);
        if (data.type === "bot") {
            addMessage(data.content, false);
        }
    
        if (data.type === "bot_audio") {
            // Base64 decode to binary
            const binaryString = atob(data.content);
            const len = binaryString.length;
            const bytes = new Uint8Array(len);
            for (let i = 0; i < len; i++) {
                bytes[i] = binaryString.charCodeAt(i);
            }
            const audioBlob = new Blob([bytes.buffer], { type: "audio/wav" });

            // Auto-play
            const audioUrl = URL.createObjectURL(audioBlob);
            const autoPlay = new Audio(audioUrl);
            autoPlay.play();

            // Add message with replay controls
            addMessage(data.text, false, false, audioBlob);
            addMessage("", false, true, audioBlob);
        }
    };

    // Initialize drag and drop
    initializeDragAndDrop();
});

// Cleanup when page unloads
window.addEventListener('beforeunload', () => {
    // Stop any playing audio and cleanup resources
    if (currentAudio) {
        currentAudio.pause();
        currentAudio = null;
    }
    
    audioInstances.forEach(audio => {
        audio.pause();
        audio.src = '';
    });
    audioInstances.clear();
    
    // Cleanup recording
    if (mediaRecorder && mediaRecorder.state !== 'inactive') {
        mediaRecorder.stop();
    }
    
    // Reset body overflow
    document.body.style.overflow = '';
});

// Notification System
class NotificationSystem {
    constructor() {
        this.container = document.getElementById('notificationContainer');
    }

    show(message, duration = 3000) {
        const notification = document.createElement('div');
        notification.className = 'notification';
        
        notification.innerHTML = `
            <div class="notification-icon">⚠️</div>
            <div>${message}</div>
            <div class="notification-close">&times;</div>
        `;

        this.container.appendChild(notification);

        setTimeout(() => {
            notification.classList.add('show');
        }, 100);

        const timeout = setTimeout(() => {
            this.hide(notification);
        }, duration);

        const closeButton = notification.querySelector('.notification-close');
        closeButton.addEventListener('click', () => {
            clearTimeout(timeout);
            this.hide(notification);
        });
    }

    hide(notification) {
        notification.classList.remove('show');
        setTimeout(() => {
            if (notification.parentNode) {
                notification.parentNode.removeChild(notification);
            }
        }, 300);
    }
}

// Initialize notification system
const notificationSystem = new NotificationSystem();

function showError(message) {
    notificationSystem.show(message, 5000);
}

function getSelectedDevice() {
    const selectElement = document.getElementById('deviceSelect');
    return selectElement.value;
}

// File input event listener
fileInput.addEventListener('change', function(e) {
    const files = e.target.files;
    
    if (files.length > 0) {
        // Add files to chat
        addFileMessage(files, true);
        
        // Show temporary file info popup
        fileInfo.style.display = 'block';
        fileInfo.classList.add('show');
        fileList.innerHTML = '';
        
        Array.from(files).forEach((file, index) => {
            const fileType = file.name.toLowerCase().endsWith('.csv') ? 'CSV' : 'MP3';
            const fileSize = (file.size / 1024 / 1024).toFixed(2);
            
            const fileItem = document.createElement('div');
            fileItem.style.marginBottom = '6px';
            fileItem.innerHTML = `
                <span class="file-name">${file.name}</span> 
                (<span class="file-type">${fileType}</span>, ${fileSize} MB)
            `;
            fileList.appendChild(fileItem);
        });

        // Process the files with spinner
        handleFileUploadWithSpinner(files);

        // Auto-hide popup after 3 seconds
        setTimeout(() => {
            hideFileInfo();
        }, 3000);

        // Clear the input for future uploads
        fileInput.value = '';
    }
});

// Updated send button handler to include file upload
sendButton.addEventListener('click', function() {
    // Send text message if there's text
    const text = messageInput.value.trim();
    if (text) {
        sendMessage();
    }
    
    // Process uploaded files if any
    if (fileInput.files.length > 0) {
        addFileMessage(fileInput.files, true);
        handleFileUploadWithSpinner(fileInput.files);
        fileInput.value = '';
    }
});

// Drag and drop support
function initializeDragAndDrop() {
    const dropZone = chatMessages;
    
    dropZone.addEventListener('dragover', (e) => {
        e.preventDefault();
        dropZone.style.backgroundColor = 'rgba(16, 185, 129, 0.1)';
    });
    
    dropZone.addEventListener('dragleave', (e) => {
        e.preventDefault();
        dropZone.style.backgroundColor = '';
    });
    
    dropZone.addEventListener('drop', (e) => {
        e.preventDefault();
        dropZone.style.backgroundColor = '';
        
        const files = Array.from(e.dataTransfer.files).filter(file => 
            file.name.toLowerCase().endsWith('.csv') || 
            file.name.toLowerCase().endsWith('.mp3')
        );
        
        if (files.length > 0) {
            const dt = new DataTransfer();
            files.forEach(file => dt.items.add(file));
            fileInput.files = dt.files;
            
            fileInput.dispatchEvent(new Event('change'));
        }
    });
}

// Click outside to hide file info
document.addEventListener('click', function(e) {
    if (!e.target.closest('.upload-button') && !e.target.closest('.file-info')) {
        hideFileInfo();
    }
});

// Progress message system (if needed)
function addProgressMessage(taskName, isUser = false) {
    const messageDiv = document.createElement('div');
    const messageId = 'progress-' + Date.now() + '-' + Math.random().toString(36).substr(2, 9);
    messageDiv.className = `message ${isUser ? 'user-message' : 'bot-message'}`;
    messageDiv.id = messageId;
    
    const currentTime = new Date().toLocaleTimeString([], {hour: '2-digit', minute:'2-digit'});
    
    messageDiv.innerHTML = `
        <div class="message-avatar">
            <div class="avatar-icon">${isUser ? '👤' : '🤖'}</div>
        </div>
        <div class="message-content">
            <div class="progress-message">
                <div class="progress-header">
                    <div class="progress-title">${taskName}</div>
                    <button class="progress-stop-btn" onclick="stopProgress('${messageId}')">
                        <svg width="14" height="14" viewBox="0 0 14 14" fill="currentColor">
                            <rect x="2" y="2" width="10" height="10" rx="1"/>
                        </svg>
                        Stop
                    </button>
                </div>
                <div class="progress-status" id="status-${messageId}">Initializing...</div>
                <div class="progress-bar-container">
                    <div class="progress-bar" id="progress-bar-${messageId}">
                        <div class="progress-fill" id="progress-fill-${messageId}" style="width: 0%"></div>
                    </div>
                    <div class="progress-percentage" id="progress-percent-${messageId}">0%</div>
                </div>
                <div class="progress-details" id="details-${messageId}">Starting process...</div>
            </div>
            <div class="message-time">${currentTime}</div>
        </div>
    `;
    
    chatMessages.appendChild(messageDiv);
    chatMessages.scrollTop = chatMessages.scrollHeight;
    
    activeProgressMessages.set(messageId, {
        element: messageDiv,
        taskName: taskName,
        startTime: Date.now(),
        progress: 0,
        status: 'running'
    });
    
    return messageId;
}

// Test functions for debugging
function testBotSpinner() {
    const spinnerId = addBotSpinner("Testing spinner system", "spinner");
    
    setTimeout(() => {
        updateBotSpinner(spinnerId, "Still processing");
    }, 2000);
    
    setTimeout(() => {
        removeBotSpinner(spinnerId, "Spinner test completed!","bot");
    }, 4000);
    
    return spinnerId;
}

console.log('🤖 Chat system with spinner loaded');
console.log('Available functions:');
console.log('- testBotSpinner() - Test spinner system');
console.log('- removeAllSpinners() - Clear all active spinners');

// Add to window for console testing
window.testBotSpinner = testBotSpinner;
window.removeAllSpinners = removeAllSpinners;