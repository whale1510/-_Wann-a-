from flask import Blueprint, render_template, request, redirect, url_for 
from pydub import AudioSegment #pip install 필요
from flask_cors import cross_origin #안전성 검사. pip 필요.
import os
import torch
import timm
import numpy as np
import os, csv
import librosa
import soundfile as sf
from .models import ASTModel
import torchaudio
import parselmouth
import pandas as pd
from parselmouth.praat import call

temp = 0

bp = Blueprint('main', __name__, url_prefix='/')

@bp.route('/', methods=['GET', 'POST'])
def index():
    current_url = request.url  # 현재 페이지의 URL 얻기
    return render_template("codeep.html")

@bp.route('/result')
def result_file():
    result = temp
    f1 = f1_value
    f2 = f2_value
    f1_2 = f1_value2
    f2_2 = f2_value2
    return render_template('result.html', result = result, data1=f1, data2=f2, data3=f1_2, data4=f2_2)

@bp.route('/upload', methods=['POST'])
def upload_file():
    print('post well')
    if 'audio' in request.files:
        audio_file = request.files['audio']
        temp_path = "received_audio.webm"  # post에서 받아오는 형태
        audio_file.save(temp_path)
        
        #아래코드가 돌아가기 위해서는, ffmpeg를 웹에서 다운로드 받아야 합니다.
        audio = AudioSegment.from_file(temp_path, format="webm") #형태를 wav로 변환
        file_path = "received_audio.wav"
        audio.export(file_path, format="wav")
        resample_audio(file_path, file_path,target_sr=16000)
        os.remove(temp_path)
        global temp, f1_value, f2_value, pitch_value, f1_value2, f2_value2
        #temp = 1 #테스트용
        temp= process_audio(file_path)

        Sound = parselmouth.Sound(file_path)
        formants_value = ['F1', 'F2']
        formant = call(Sound, "To Formant (burg)", 0.005, 5.5, 4700, 0.025, 50)
        df = pd.DataFrame({"times": formant.ts()})
        pitch = Sound.to_pitch()
        for idx, col in enumerate(formants_value, 1):
            df[col] = df['times'].map(lambda x: formant.get_value_at_time(formant_number=idx, time=x))
        df['F0(pitch)'] = df['times'].map(lambda x: pitch.get_value_at_time(time=x))
        df = df.dropna()
        df = remove_out(df, ['F1', 'F2'])
        f1_value =most_appear(df['F1'])
        f2_value = most_appear(df['F2'])
        pitch_value = most_appear(df['F0(pitch)'])

        file_path2 = "a.wav"
        Sound2 = parselmouth.Sound(file_path2)
        formants_value2 = ['F1', 'F2']
        formant2 = call(Sound2, "To Formant (burg)", 0.005, 5.5, 4700, 0.025, 50)
        df2 = pd.DataFrame({"times": formant2.ts()})
        pitch2 = Sound2.to_pitch()
        for idx2, col2 in enumerate(formants_value2, 1):
            df2[col2] = df2['times'].map(lambda x: formant2.get_value_at_time(formant_number=idx2, time=x))
        df2['F0(pitch)'] = df2['times'].map(lambda x: pitch2.get_value_at_time(time=x))
        df2 = df2.dropna()
        df2 = remove_out(df2, ['F1', 'F2'])
        f1_value2 =most_appear(df2['F1'])
        f2_value2 = most_appear(df2['F2'])
        return "result"
    else:
        return {'status': 'no file'}
    
class ASTModelVis(ASTModel):
    def get_att_map(self, block, x):
        qkv = block.attn.qkv
        num_heads = block.attn.num_heads
        scale = block.attn.scale
        B, N, C = x.shape
        qkv = qkv(x).reshape(B, N, 3, num_heads, C // num_heads).permute(2, 0, 3, 1, 4)
        q, k, v = qkv[0], qkv[1], qkv[2]
        attn = (q @ k.transpose(-2, -1)) * scale
        attn = attn.softmax(dim=-1)
        return attn

    def forward_visualization(self, x):
        x = x.unsqueeze(1)
        x = x.transpose(2, 3)

        B = x.shape[0]
        x = self.v.patch_embed(x)
        cls_tokens = self.v.cls_token.expand(B, -1, -1)
        dist_token = self.v.dist_token.expand(B, -1, -1)
        x = torch.cat((cls_tokens, dist_token, x), dim=1)
        x = x + self.v.pos_embed
        x = self.v.pos_drop(x)
        # save the attention map of each of 12 Transformer layer
        att_list = []
        for blk in self.v.blocks:
            cur_att = self.get_att_map(blk, x)
            att_list.append(cur_att)
            x = blk(x)
        return att_list
def remove_out(dataframe, remove_col):
    dff = dataframe
    for k in remove_col:
        level_1q = dff[k].quantile(0.25)
        level_3q = dff[k].quantile(0.75)
        IQR = level_3q - level_1q
        rev_range = 2
        dff = dff[(dff[k] <= level_3q + (rev_range * IQR)) & (dff[k] >= level_1q - (rev_range * IQR))]
        dff = dff.reset_index(drop = True)
    return dff
def most_appear(df):
    data = df
    data_array = np.array(data)
    hist, bin_edges = np.histogram(data_array, bins='auto')
    most_common_bin_index = np.argmax(hist)
    most_common_bin_start = bin_edges[most_common_bin_index]
    most_common_bin_end = bin_edges[most_common_bin_index + 1]
    # print("최빈 구간:", most_common_bin_start, "-", most_common_bin_end)
    # print("최빈 구간:", (most_common_bin_start + most_common_bin_end)/2)
    most_common_values = data_array[(data_array >= most_common_bin_start) & (data_array <= most_common_bin_end)]
    return most_common_values.mean()

def make_features(wav_name, mel_bins, target_length=1024):
    waveform, sr = torchaudio.load(wav_name)
    assert sr == 16000, 'input audio sampling rate must be 16kHz'

    fbank = torchaudio.compliance.kaldi.fbank(
        waveform, htk_compat=True, sample_frequency=sr, use_energy=False,
        window_type='hanning', num_mel_bins=mel_bins, dither=0.0, frame_shift=10)

    n_frames = fbank.shape[0]

    p = target_length - n_frames
    if p > 0:
        m = torch.nn.ZeroPad2d((0, 0, 0, p))
        fbank = m(fbank)
    elif p < 0:
        fbank = fbank[0:target_length, :]

    fbank = (fbank - (-4.2677393)) / (4.5689974 * 2)
    return fbank    

def resample_audio(input_file, output_file, target_sr=16000):
    try:
        audio, sr = librosa.load(input_file, sr=None)
        resampled_audio = librosa.resample(audio, orig_sr=sr, target_sr=target_sr)
        target_duration = 10.0
        repeated_audio = []
        while len(repeated_audio) < sr * target_duration:
            repeated_audio.extend(resampled_audio)
        repeated_audio = np.array(repeated_audio[:int(sr * target_duration)])
        sf.write(output_file, repeated_audio, target_sr)
    except Exception as e:
        print(f"오류 발생: {e}")


def process_audio(file_path):
    feats = make_features(file_path, mel_bins=128) 
    feats_data = feats.expand(1, 1024, 128) # feature을 모델의 입력에 알맞게 맞춰준다
    feats_data = feats_data.to(torch.device("cpu"))
                               
    current_directory = os.getcwd()
    input_tdim = 1024
    current_directory = os.getcwd()
    checkpoint_path1 = os.path.join(current_directory, 'best_audio_modela.pth')
    checkpoint_path2 = os.path.join(current_directory, 'best_audio_modele.pth')
    
    ast_mdl = ASTModelVis(label_dim=2, input_tdim=input_tdim, imagenet_pretrain=False, audioset_pretrain=False) # 모델 class 정의
    checkpoint = torch.load(checkpoint_path1, map_location='cpu') # check point 로드
    audio_model = torch.nn.DataParallel(ast_mdl, device_ids=[0]) # 모델 로드
    audio_model.load_state_dict(checkpoint)  # audio 모델에 checkpoint 얹기
    audio_model = audio_model.to(torch.device("cpu")) # 모델을 cpu 디바이스로 예측하도록 설정
    audio_model.eval()
    with torch.no_grad():
    #   with autocast():
        output = audio_model.forward(feats_data)
        output = torch.sigmoid(output)
    result_output = output.data.cpu().numpy()[0]
    sorted_indexes = np.argsort(result_output)[::-1]

    result = (sorted_indexes[0])
    return result
