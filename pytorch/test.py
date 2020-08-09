import os
import sys
sys.path.insert(1, os.path.join(sys.path[0], '../utils'))
import numpy as np
import argparse
import librosa
import matplotlib.pyplot as plt
import torch

from utilities import create_folder, get_filename
from models import *
from pytorch_utils import move_data_to_device
import config


def audio_tagging(args):
    """Inference audio tagging result of an audio clip.
    """

    # Arugments & parameters
    window_size = args.window_size
    hop_size = args.hop_size
    mel_bins = args.mel_bins
    fmin = args.fmin
    fmax = args.fmax
    model_type = args.model_type
    audio_path = args.audio_path
    
    device = torch.device('cuda') if args.cuda and torch.cuda.is_available() else torch.device('cpu')

    sample_rate = config.sample_rate
    classes_num = config.classes_num
    labels = config.labels

    # Model
    checkpoint_path = '/data/dean/panns/audioset_tagging_cnn/pytorch/Cnn10_mAP=0.380.pth'
    checkpoint_path_2 = '/data/dean/audioset_tagging_cnn/workspaces/checkpoints/main/sample_rate=32000,window_size=1024,hop_size=320,mel_bins=64,fmin=50,fmax=14000/data_type=full_train/Cnn10/loss_type=clip_bce/balanced=balanced/augmentation=none/batch_size=32/N=5,length=2/198000_iterations.pth'
    Model = eval(model_type)
    model = Model(sample_rate=sample_rate, window_size=window_size, 
        hop_size=hop_size, mel_bins=mel_bins, fmin=fmin, fmax=fmax, 
        classes_num=classes_num)
    Model_2 = eval(model_type+'_local')
    model_2 = Model_2(sample_rate=sample_rate, window_size=window_size, 
        hop_size=hop_size, mel_bins=mel_bins, fmin=fmin, fmax=fmax, 
        classes_num=classes_num, N=5, length=2)
    
    checkpoint = torch.load(checkpoint_path, map_location=device)
    model.load_state_dict(checkpoint['model'])
    
    checkpoint_2 = torch.load(checkpoint_path_2, map_location=device)
    model_2.load_state_dict(checkpoint_2['model'])

    # Parallel
    print('GPU number: {}'.format(torch.cuda.device_count()))
    model = torch.nn.DataParallel(model)
    model_2 = torch.nn.DataParallel(model_2)

    if 'cuda' in str(device):
        model.to(device)
        model_2.to(device)
    
    # Load audio
    (waveform, _) = librosa.core.load(audio_path, sr=sample_rate, mono=True)
    waveform_ = waveform
    
#     stft_ = librosa.core.stft(y=waveform,n_fft=window_size,hop_length=hop_size).T
#     melW = librosa.filters.mel(sr=sample_rate,n_fft=window_size,n_mels=mel_bins,fmin=fmin,fmax=fmax).T
#     mel_spec = np.dot(np.abs(stft_)**2,melW)
#     logmel = librosa.core.power_to_db(mel_spec,ref=1.0,amin=1e-10,top_db=None)
#     logmel = logmel.astype(np.float32)
#     logmel = np.transpose(logmel,(1,0))
#     plt.imshow(logmel,cmap=plt.cm.jet)
#     plt.axis('off')
#     fig = plt.gcf()
#     height,width=logmel.shape
#     fig.set_size_inches(width/40.,height/40.)
#     plt.gca().xaxis.set_major_locator(plt.NullLocator())
#     plt.gca().yaxis.set_major_locator(plt.NullLocator())
#     plt.subplots_adjust(top=1,bottom=0,right=1,left=0,hspace=0,wspace=0)
#     plt.margins(0,0)
#     plt.savefig('waveform.png',dpi=200,pad_inches=0)
    
    waveform = waveform[None, :]   # (1, audio_length)
    waveform = move_data_to_device(waveform, device)

    # Forward
    with torch.no_grad():
        model.eval()
        model_2.eval()
        batch_output_dict_2 = model(waveform, None)
        batch_output_dict = model_2(waveform, batch_output_dict_2['clipwise_output'], batch_output_dict_2['feature_map'], None)

    clipwise_output = batch_output_dict['prob'].data.cpu().numpy()[0]
    """(classes_num,)"""

    sorted_indexes = np.argsort(clipwise_output)[::-1]

    # Print audio tagging top probabilities
    for k in range(10):
        print('{}: {:.3f}'.format(np.array(labels)[sorted_indexes[k]], 
            clipwise_output[sorted_indexes[k]]))
        
    waveform_1 = waveform_[109395:173395]
    stft_ = librosa.core.stft(y=waveform_1,n_fft=window_size,hop_length=hop_size).T
    melW = librosa.filters.mel(sr=sample_rate,n_fft=window_size,n_mels=mel_bins,fmin=fmin,fmax=fmax).T
    mel_spec = np.dot(np.abs(stft_)**2,melW)
    logmel = librosa.core.power_to_db(mel_spec,ref=1.0,amin=1e-10,top_db=None)
    logmel = logmel.astype(np.float32)
    logmel = np.transpose(logmel,(1,0))
    plt.imshow(logmel,cmap=plt.cm.jet)
    plt.axis('off')
    fig = plt.gcf()
    height,width=logmel.shape
    fig.set_size_inches(width/40.,height/40.)
    plt.gca().xaxis.set_major_locator(plt.NullLocator())
    plt.gca().yaxis.set_major_locator(plt.NullLocator())
    plt.subplots_adjust(top=1,bottom=0,right=1,left=0,hspace=0,wspace=0)
    plt.margins(0,0)
    plt.savefig('waveform1.png',dpi=200,pad_inches=0)
    
    waveform_2 = waveform_[34976:98976]
    stft_ = librosa.core.stft(y=waveform_2,n_fft=window_size,hop_length=hop_size).T
    melW = librosa.filters.mel(sr=sample_rate,n_fft=window_size,n_mels=mel_bins,fmin=fmin,fmax=fmax).T
    mel_spec = np.dot(np.abs(stft_)**2,melW)
    logmel = librosa.core.power_to_db(mel_spec,ref=1.0,amin=1e-10,top_db=None)
    logmel = logmel.astype(np.float32)
    logmel = np.transpose(logmel,(1,0))
    plt.imshow(logmel,cmap=plt.cm.jet)
    plt.axis('off')
    fig = plt.gcf()
    height,width=logmel.shape
    fig.set_size_inches(width/40.,height/40.)
    plt.gca().xaxis.set_major_locator(plt.NullLocator())
    plt.gca().yaxis.set_major_locator(plt.NullLocator())
    plt.subplots_adjust(top=1,bottom=0,right=1,left=0,hspace=0,wspace=0)
    plt.margins(0,0)
    plt.savefig('waveform2.png',dpi=200,pad_inches=0)
    
    waveform_3 = waveform_[146604:210604]
    stft_ = librosa.core.stft(y=waveform_3,n_fft=window_size,hop_length=hop_size).T
    melW = librosa.filters.mel(sr=sample_rate,n_fft=window_size,n_mels=mel_bins,fmin=fmin,fmax=fmax).T
    mel_spec = np.dot(np.abs(stft_)**2,melW)
    logmel = librosa.core.power_to_db(mel_spec,ref=1.0,amin=1e-10,top_db=None)
    logmel = logmel.astype(np.float32)
    logmel = np.transpose(logmel,(1,0))
    plt.imshow(logmel,cmap=plt.cm.jet)
    plt.axis('off')
    fig = plt.gcf()
    height,width=logmel.shape
    fig.set_size_inches(width/40.,height/40.)
    plt.gca().xaxis.set_major_locator(plt.NullLocator())
    plt.gca().yaxis.set_major_locator(plt.NullLocator())
    plt.subplots_adjust(top=1,bottom=0,right=1,left=0,hspace=0,wspace=0)
    plt.margins(0,0)
    plt.savefig('waveform3.png',dpi=200,pad_inches=0)
    
    waveform_4 = waveform_[49860:113860]
    stft_ = librosa.core.stft(y=waveform_4,n_fft=window_size,hop_length=hop_size).T
    melW = librosa.filters.mel(sr=sample_rate,n_fft=window_size,n_mels=mel_bins,fmin=fmin,fmax=fmax).T
    mel_spec = np.dot(np.abs(stft_)**2,melW)
    logmel = librosa.core.power_to_db(mel_spec,ref=1.0,amin=1e-10,top_db=None)
    logmel = logmel.astype(np.float32)
    logmel = np.transpose(logmel,(1,0))
    plt.imshow(logmel,cmap=plt.cm.jet)
    plt.axis('off')
    fig = plt.gcf()
    height,width=logmel.shape
    fig.set_size_inches(width/40.,height/40.)
    plt.gca().xaxis.set_major_locator(plt.NullLocator())
    plt.gca().yaxis.set_major_locator(plt.NullLocator())
    plt.subplots_adjust(top=1,bottom=0,right=1,left=0,hspace=0,wspace=0)
    plt.margins(0,0)
    plt.savefig('waveform4.png',dpi=200,pad_inches=0)
    
    waveform_5 = waveform_[5209:69209]
    stft_ = librosa.core.stft(y=waveform_5,n_fft=window_size,hop_length=hop_size).T
    melW = librosa.filters.mel(sr=sample_rate,n_fft=window_size,n_mels=mel_bins,fmin=fmin,fmax=fmax).T
    mel_spec = np.dot(np.abs(stft_)**2,melW)
    logmel = librosa.core.power_to_db(mel_spec,ref=1.0,amin=1e-10,top_db=None)
    logmel = logmel.astype(np.float32)
    logmel = np.transpose(logmel,(1,0))
    plt.imshow(logmel,cmap=plt.cm.jet)
    plt.axis('off')
    fig = plt.gcf()
    height,width=logmel.shape
    fig.set_size_inches(width/40.,height/40.)
    plt.gca().xaxis.set_major_locator(plt.NullLocator())
    plt.gca().yaxis.set_major_locator(plt.NullLocator())
    plt.subplots_adjust(top=1,bottom=0,right=1,left=0,hspace=0,wspace=0)
    plt.margins(0,0)
    plt.savefig('waveform5.png',dpi=200,pad_inches=0)
    
    return clipwise_output, labels



if __name__ == '__main__':

    parser = argparse.ArgumentParser(description='Example of parser. ')
    subparsers = parser.add_subparsers(dest='mode')

    parser_at = subparsers.add_parser('audio_tagging')
    parser_at.add_argument('--window_size', type=int, default=1024)
    parser_at.add_argument('--hop_size', type=int, default=320)
    parser_at.add_argument('--mel_bins', type=int, default=64)
    parser_at.add_argument('--fmin', type=int, default=50)
    parser_at.add_argument('--fmax', type=int, default=14000) 
    parser_at.add_argument('--model_type', type=str, required=True)
    parser_at.add_argument('--audio_path', type=str, required=True)
    parser_at.add_argument('--cuda', action='store_true', default=False)  
    args = parser.parse_args()

    if args.mode == 'audio_tagging':
        audio_tagging(args)
    else:
        raise Exception('Error argument!')