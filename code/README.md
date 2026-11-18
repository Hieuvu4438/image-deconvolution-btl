# Chạy lại thực nghiệm Deconvolution

Chạy từ thư mục gốc `BTL`. Môi trường đã kiểm chứng là Python 3.13.5, Windows 11, JDK 25.0.1, ImageJ 1.54p và DeconvolutionLab2 2.1.2. Dùng môi trường Python sẵn có với các gói trong [requirements.txt](requirements.txt). Lần đầu cần `curl` và mạng để lấy archive Berkeley cùng hai JAR chính thức. Script kiểm tra SHA-256 của JAR và ghi hash của dataset, từng ảnh JPEG.

```powershell
python code/scripts/prepare_data.py
python code/scripts/check.py
python code/scripts/run_experiments.py
python code/scripts/audit_artifacts.py
python code/scripts/run_imagej.py
python code/scripts/run_sensitivity.py
python code/scripts/run_epfl_bars.py
python code/scripts/build_report.py
```

`run_experiments.py` tạo một thư mục mới và cập nhật `outputs/LATEST.txt` khi phần Python chạy xong. Các lệnh tiếp theo lấy thư mục này. `run_imagej.py`, `run_sensitivity.py` và `build_report.py` cũng hỗ trợ `--run <đường-dẫn>`. ImageJ và thử nghiệm độ nhạy tạo thư mục đầu ra riêng cho từng lần chạy để giữ lại bằng chứng gốc. Khi dùng một lần chạy khác để đưa vào báo cáo, đối chiếu phần diễn giải với số mới, đặc biệt phần thời gian benchmark.

## Minh họa trên giao diện ImageJ hoặc Fiji

1. Tải đúng JAR v2.1.2 vào thư mục `plugins` và khởi động lại. Vào **Plugins → DeconvolutionLab2 → DeconvolutionLab2 Lab**. [Tài liệu EPFL](https://bigwww.epfl.ch/deconvolution/deconvolutionlab2/).
2. Đối với `100007_poisson`, dùng `test/100007_poisson/observed.tif` và `imagej/100007_poisson/psf_centered.tif` trong thư mục lần chạy. PSF cho plugin là lưới 256 × 256 đặt tâm ở pixel (128,128), chỉ số từ 0. `test/100007_poisson/psf.tif` là kernel nhỏ của Python.
3. Chọn TRIF với λ = 0,02 để tái hiện một ca. Dùng Academic FFT, chuẩn hóa tổng PSF bằng 1, padding và apodization tắt, constraint tắt. Đầu ra chọn stack, float, intact. Dùng `reference.tif` của cùng ca để tính điểm.
4. [BatchReference.java](imagej/BatchReference.java) gọi chính điểm vào `DeconvolutionLab2_Run` bằng lệnh macro trong runtime ImageJ. Lệnh đầy đủ nằm trong `imagej/plugin.log`. `-stats false` và `noshow` dùng cho batch không giao diện, còn Java đọc lại TIFF để chấm ba metric. Giá trị `dl2_psnr_raw` đến từ `signal.Assessment.psnr` của plugin và dùng đỉnh theo ảnh tham chiếu.

## Đọc dữ liệu và số đo

| Tệp trong thư mục lần chạy | Nội dung |
|---|---|
| `manifest.json`, `config_frozen_before_test.json` | Nguồn dữ liệu, ID, SHA-256, lưới và cấu hình khóa |
| `validation_sweep.csv`, `test_metrics.csv` | Chọn tham số theo 5 ảnh validation và chấm test với cấu hình cố định |
| `test/<id>_<noise>/` | TIFF float32 tham chiếu, ảnh mờ, ảnh quan sát, PSF, đầu ra raw và hình PNG |
| `imagej/jobs.tsv`, `imagej/plugin.log` | Các ca và lệnh thực chạy với plugin |
| `imagej/imagej_metrics.csv`, `imagej/parity.csv` | MSE, PSNR, SSIM, native PSNR và sai khác pixel gốc |
| `sensitivity/` | Sáu kịch bản sai mô hình hoặc đối chứng, phantom, CSV và đường cắt |
| `epfl-bars/` | Phép thử độc lập trên hình chiếu tổng trục z từ dữ liệu EPFL, gồm TIFF và điểm của Python lẫn ImageJ |
| `artifact_audit.json`, `imagej/audit.json` | Kiểm tra tính đủ, metric từ TIFF và đối chiếu theo ngưỡng |

MSE và PSNR dùng mọi pixel, đỉnh cố định bằng 1. SSIM lấy trung bình các cửa sổ Gaussian hợp lệ 11 × 11 với σ = 1,5 và thống kê có trọng số theo tổng thể. CSV có cả kết quả raw và cắt về [0,1]. Dùng PNG để xem và TIFF raw để đối chiếu số.
