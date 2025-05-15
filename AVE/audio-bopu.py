import numpy as np
import matplotlib.pyplot as plt
import torch
from PIL import Image


def create_spectrogram(audio, sr=48000, n_fft=1024, hop=512, vmin=-100, vmax=0):
    # 确保音频是 torch.Tensor 类型
    audio = torch.as_tensor(audio)

    # 应用短时傅里叶变换(STFT)
    window = torch.hann_window(n_fft)
    spec = torch.stft(audio, n_fft, hop, window=window, return_complex=True)

    # 计算幅度谱并转换为分贝刻度
    spec = spec.abs().clamp_min(1e-12).log10().mul(10)

    # 如果是多声道,取平均
    if spec.dim() > 2:
        spec = spec.mean(0)

    # 将谱图转换为numpy数组
    spec_np = spec.cpu().numpy()

    # 计算时间和频率轴
    t = np.arange(0, spec_np.shape[1]) * hop / sr
    f = np.arange(0, spec_np.shape[0]) * sr / 2 / (n_fft // 2) / 1000

    # 创建图形
    fig, ax = plt.subplots(figsize=(15, 5))

    # 绘制谱图
    im = ax.pcolormesh(t, f, spec_np, shading='auto', vmin=vmin, vmax=vmax, cmap='inferno')

    # 设置标签
    ax.set_xlabel('Time [s]')
    ax.set_ylabel('Frequency [kHz]')
    ax.set_title('Spectrogram')

    # 添加颜色条
    plt.colorbar(im, ax=ax, format='%+2.0f dB')

    # 调整布局
    plt.tight_layout()

    # 将图形转换为PIL图像
    fig.canvas.draw()
    img = Image.frombytes('RGB', fig.canvas.get_width_height(), fig.canvas.tostring_rgb())

    # 关闭matplotlib图形以释放内存
    plt.close(fig)

    return img


# 读取 WAV 文件
sr, audio = wavfile.read(r"E:\allchat\output_16000_mono_0.wav")

# 如果音频是整数类型，将其转换为 -1.0 到 1.0 之间的浮点数
if audio.dtype == np.int16:
    audio = audio.astype(np.float32) / 32768.0
elif audio.dtype == np.int32:
    audio = audio.astype(np.float32) / 2147483648.0

# 如果音频是立体声，取平均得到单声道
if audio.ndim > 1:
    audio = audio.mean(axis=1)

# 创建谱图
spectrogram = create_spectrogram(audio, sr=sr)

# 显示图像
spectrogram.show()

# 保存图像
spectrogram.save('1_spectrogram.png')
