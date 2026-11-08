# ASR-MDD: Phát hiện phát âm sai bằng CTC

## Giới thiệu đề tài

ASR-MDD (Automatic Speech Recognition for Mispronunciation Detection) là bài toán nhận diện và phát hiện lỗi phát âm trong lời nói. Với mỗi mẫu âm thanh, hệ thống nhận thêm chuỗi âm vị chuẩn (*canonical phonemes*) và dự đoán chuỗi âm vị người nói thực sự phát ra (*transcript phonemes*).

Mục tiêu không chỉ là chuyển giọng nói thành chuỗi âm vị, mà còn nhận ra sự khác biệt giữa cách phát âm chuẩn và cách phát âm trong bản ghi. Ví dụ, nếu chuỗi chuẩn là `a b c` nhưng người nói phát âm thành `a d c`, hệ thống cần dự đoán chuỗi `a d c` để lỗi thay thế âm vị có thể được đánh giá.

Dự án sử dụng đặc trưng âm thanh từ mô hình Wav2Vec2 tiếng Việt và mô hình Transformer kết hợp thông tin âm thanh với chuỗi âm vị chuẩn. Đầu ra được giải mã bằng CTC (Connectionist Temporal Classification), phù hợp với bài toán căn chỉnh giữa các khung âm thanh và chuỗi âm vị khi không có nhãn thời gian cho từng âm.

## Pipeline thực hiện

1. **Đọc và ghép dữ liệu:** đọc CSV thông tin mẫu và CSV âm vị, ghép theo `id`,   ánh xạ đường dẫn đến file WAV và kiểm tra file âm thanh tồn tại.
2. **Tạo từ vựng âm vị:** tổng hợp các âm vị từ tập huấn luyện và tập public test. ID `0` được dành cho ký hiệu blank của CTC.
3. **Trích xuất đặc trưng âm thanh:** chuyển âm thanh về mono và tần số lấy mẫu 16 kHz, sau đó dùng Wav2Vec2 để lấy chuỗi embedding theo thời gian. Wav2Vec2 được giữ đóng băng; các embedding được tạo dưới `no_grad`.
4. **Tăng cường và tạo batch:** trong huấn luyện, âm thanh có thể được thay đổi tốc độ nhẹ; SpecAugment che một số vùng theo thời gian và chiều đặc trưng.  WeightedRandomSampler tăng cơ hội lấy các mẫu phát âm sai.
5. **Kết hợp âm thanh và âm vị chuẩn:** mô hình chiếu đặc trưng âm thanh vào không gian ẩn, mã hóa chuỗi âm vị chuẩn, rồi dùng attention để kết hợp hai  nguồn thông tin. CTC head dự đoán phân phối âm vị trên từng khung.
6. **Huấn luyện:** tối ưu tổng loss gồm CTC trên transcript, thành phần phạt dự đoán trùng chuỗi chuẩn và contrastive loss giữa biểu diễn âm thanh với biểu diễn âm vị chuẩn. Lưu checkpoint có validation loss tốt nhất và dừng sớm khi không còn cải thiện.
7. **Suy luận và đánh giá:** giải mã CTC theo greedy decoding, ghi dự đoán ra CSV. Nếu có nhãn public test, tính F1 phát hiện lỗi, PER (Phoneme Error Rate), DER (Detection Error Rate) và điểm tổng hợp:

   `score = 0.5 × F1 + 0.4 × (1 − DER) + 0.1 × (1 − PER)`

## Cấu trúc mã nguồn

```text
src/mdd_challen/
├── application.py  # Điều phối các bước của ứng dụng
├── config.py       # Cấu hình dữ liệu, mô hình và huấn luyện
├── data.py         # Đọc dữ liệu, từ vựng, trích đặc trưng và DataLoader
├── losses.py       # CTC, canonical penalty và contrastive loss
├── metrics.py      # Căn chỉnh âm vị, F1, PER và DER
├── model.py        # SpecAugment, Transformer và CTC head
└── training.py     # Huấn luyện, checkpoint, decoder và suy luận
```

Notebook tại `notebooks/ctc-mddv2.ipynb` là ví dụ gọi package. Có thể chạy
pipeline trực tiếp bằng CLI mà không cần notebook.

## Cài đặt

Ví dụ trên Windows PowerShell:

```powershell
python -m venv .venv
.\.venv\Scripts\Activate.ps1
python -m pip install -r requirements.txt
python -m pip install -e ".[dev]"
```

Để chạy bằng GPU, cài phiên bản PyTorch và TorchAudio tương thích với phiên bản
CUDA trên máy theo hướng dẫn cài đặt chính thức của PyTorch.

## Chuẩn bị dữ liệu

Đưa các CSV vào cùng một thư mục dữ liệu:

```text
data/
├── train.csv
├── train_phones.csv
├── public_test.csv
└── public_test_phones.csv
```

- `train.csv` và `public_test.csv` cần có các cột `id`, `path`.
- `train_phones.csv` và `public_test_phones.csv` cần có `id`, `canonical`,
  `transcript`. Có thể dùng tên tương đương `canonical_phoneme` và
  `transcript_phoneme`.
- Các giá trị `path` trỏ đến tên file WAV. File WAV được tìm trong thư mục âm
  thanh tương ứng; nếu giá trị chưa có đuôi `.wav`, chương trình tự thêm đuôi.
- Thư mục âm thanh train và public test được truyền riêng bằng CLI.

Ví dụ các cột:

```csv
id,path
sample_001,sample_001.wav
```

```csv
id,canonical,transcript
sample_001,a b c,a d c
```

Trong đó chuỗi âm vị được phân tách bằng dấu cách. Tập train được chia train và
validation theo tỷ lệ mặc định 90/10, có stratify theo nhãn phát âm sai.

## Huấn luyện và đánh giá

Chạy từ thư mục gốc repository:

```powershell
python -m mdd_challen `
  --data-dir "D:\datasets\mdd\data" `
  --audio-dir "D:\datasets\mdd\train_audio" `
  --test-audio-dir "D:\datasets\mdd\public_test_audio" `
  --checkpoint-dir checkpoints `
  --output-dir outputs `
  --epochs 50 `
  --batch-size 8
```

Mặc định mô hình chọn GPU nếu khả dụng, nếu không thì chạy trên CPU. Có thể chỉ
định thiết bị bằng `--device cpu` hoặc `--device cuda`. Tên mô hình Wav2Vec2 có
thể thay bằng `--pretrained-model`.

Kết quả được lưu như sau:

- `checkpoints/best_model.pt`: trọng số mô hình tốt nhất trên validation.
- `checkpoints/train_history.csv`: loss train/validation theo epoch.
- `checkpoints/phoneme_vocab.json`: ánh xạ âm vị sang ID.
- `outputs/results.csv`: dự đoán cho public test.

Khi `public_test_phones.csv` có mặt, chương trình tự tính và in F1, DER, PER và
điểm tổng hợp sau khi suy luận.

### Chỉ suy luận từ checkpoint

Để bỏ qua huấn luyện và dùng checkpoint có sẵn:

```powershell
python -m mdd_challen `
  --data-dir "D:\datasets\mdd\data" `
  --audio-dir "D:\datasets\mdd\train_audio" `
  --test-audio-dir "D:\datasets\mdd\public_test_audio" `
  --checkpoint checkpoints\best_model.pt `
  --checkpoint-dir checkpoints `
  --output-dir outputs
```

Từ vựng được tạo lại từ các CSV đầu vào; từ vựng này phải khớp checkpoint.
Chương trình báo lỗi nếu ánh xạ âm vị không khớp.

## Chạy kiểm thử

```powershell
python -m pytest
```

Các kiểm thử hiện có bao phủ từ vựng và giải mã CTC, padding batch, ghép CSV,
forward/loss của mô hình và các metric trên dữ liệu mẫu. Việc huấn luyện đầy đủ
cần bộ dữ liệu WAV và có thể cần GPU; repository không kèm dữ liệu này.
