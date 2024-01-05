// script.js

let audioChunks = [];
let mediaRecorder;
let audioBlob;
let audioUrl;
let audio = new Audio();

function toggleRecordingModal(show) {
    document.getElementById('recordingModal').style.display = show ? 'block' : 'none';
    if (show) {
        // 음파 모션 초기화
        initializeWaveform();
    }
}

function startRecording() {
    navigator.mediaDevices.getUserMedia({ audio: true })
        .then(stream => {
            mediaRecorder = new MediaRecorder(stream);
            audioChunks = [];
            mediaRecorder.start();
            toggleRecordingModal(true);

            mediaRecorder.addEventListener("dataavailable", event => {
                audioChunks.push(event.data);
            });

            mediaRecorder.addEventListener("stop", () => {
                audioBlob = new Blob(audioChunks);
                //audioBlob = new Blob(audioChunks, { 'type' : 'audio/wav' }); // Blob 타입을 audio/wav로 설정

                audioUrl = URL.createObjectURL(audioBlob);
                audio.src = audioUrl;
            });

            // 실시간으로 오디오 데이터를 받아와서 음파 모션 처리
            //updateWaveform(stream);
        });
}

function stopRecording() {
    mediaRecorder.stop();
    toggleRecordingModal(false);
    sendAudioToServer();
}

function initializeWaveform() {
    // 음파 모션 초기화 로직
}

function updateWaveform(stream) {
    // 실시간 오디오 데이터를 이용한 음파 모션 업데이트 로직
}

document.getElementById('recordButton').addEventListener('click', function() {startRecording()});
document.getElementById('submitButton').addEventListener('click', function() {stopRecording();});

//추가, 서버로 post하는 과정
function sendAudioToServer() {
    const formData = new FormData();
    formData.append("audio", audioBlob, "recording.wav");

    fetch("http://localhost:5000/upload", { // Flask 서버의 URL과 엔드포인트
        method: "POST",
        body: formData
    })
    .then(response => response.json())
    .then(data => {
        console.log("Success:", data);
    })
    .catch((error) => {
        console.error("Error:", error);
    });
}

mediaRecorder.addEventListener("stop", () => {
    audioBlob = new Blob(audioChunks, { 'type' : 'audio/wav' }); // Blob 타입을 audio/wav로 설정
    audioUrl = URL.createObjectURL(audioBlob);
    audio.src = audioUrl;

    sendAudioToServer(); // 녹음이 끝나면 서버에 오디오 파일 전송
});