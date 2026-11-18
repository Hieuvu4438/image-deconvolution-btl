# Kế hoạch triển khai TV2 — BSDS500

Ngày 18/11/2026 — Phạm Mạnh Dũng. Thực hiện theo gate, thay thế các mốc giờ đã trôi qua và phạm vi DIP3E/benchmark cố định trong file 02.

## Thiết kế chốt trước khi chạy

- BSDS500 chính thức: https://www2.eecs.berkeley.edu/Research/Projects/CS/vision/grouping/resources.html . Dùng JPEG làm ảnh tham chiếu, không dùng nhãn phân đoạn.
- Chọn 5 ảnh validation và 10 ảnh test theo tên tăng dần, crop tâm 256×256, chuyển xám; lưu manifest/SHA256. Chỉ chọn tham số trên validation, khóa config trước test.
- PSF Gaussian 17×17, sigma=1.9, tổng=1. Biên tuần hoàn, float64, thang [0,1]. Gaussian sigma=3/255 (cắt âm về 0 để RL hợp lệ); Poisson peak=700; seed cố định.
- Cài NIF/TRIF/RIF/RL theo TV1. RLTV: công thức (21) chỉ là TV; bổ sung bước cập nhật từ Sage et al. mục 3.9 và ghi rõ biến thể sai phân.
- Báo cáo cả raw và clipped metrics; SSIM Gaussian 11×11, sigma=1.5, covariance tổng thể, vùng cửa sổ hợp lệ. Không hứa PSNR/tốc độ trước khi đo.
- Không tái lập bảng synthetic của TV1 khi chưa có ảnh/code gốc. PSF chẵn chỉ là unit test riêng.

## Phase và bằng chứng

| Phase | Công việc                                            | Gate và cập nhật báo cáo                                   |
| ----- | ---------------------------------------------------- | ---------------------------------------------------------- |
| P0    | Kế hoạch và khung báo cáo                            | Báo cáo 04 có ba mục bắt buộc                              |
| P1    | Tải dữ liệu, manifest, I/O, PSF                      | 15 ảnh; checksum, nguồn, quy ước vào mục 1                 |
| P2    | 5 solver NumPy/SciPy, metrics; baseline và optimized | Công thức và mã vào mục 1, thiết kế tối ưu vào mục 2       |
| P3    | Unit tests tính tay, toán tử độc lập, I/O, metric    | unittest PASS, log và giải thích vào mục 3                 |
| P4    | Sweep val, khóa config, test, benchmark              | CSV, TIFF raw, PNG, hình so sánh; mục 2 và 4 có số thực    |
| P5    | Tổng hợp, gallery, bàn giao Fiji                     | Kiểm tra artifact/link; mục 5 có lệnh tái chạy và giới hạn |

Sau mỗi phase cập nhật [báo cáo TV2](04_bao_cao_phan_thanh_vien_2.md) trước khi tiếp tục.

## Tái chạy từ gốc repo

```powershell
.\.venv\Scripts\python.exe team-project/code/scripts/prepare_data.py
.\.venv\Scripts\python.exe -m unittest discover -s team-project/code/tests -v
.\.venv\Scripts\python.exe team-project/code/scripts/run_experiments.py
```

Dataset trong team-project/data/bsds500; mỗi lần chạy tạo thư mục riêng trong team-project/outputs. Giữ dependency hiện có. Đo median thời gian sau warmup, bộ nhớ tracemalloc là cấp phát được theo dõi, không phải peak RSS và không so trực tiếp với Java.

## Đối chiếu Fiji

Bàn giao cùng reference.tif, observed.tif, psf.tif và config. TV3 phải ghi phiên bản plugin, padding, apodization, chuẩn hóa PSF, constraint, epsilon, K/lambda và ảnh xuất thật. Chưa có output Fiji thì ghi pending. Không tự gửi tin nhắn hoặc cập nhật Google Docs.

## Trạng thái sau thực thi

- [x] P0: kế hoạch và báo cáo riêng.
- [x] P1: 15 ảnh BSDS500, manifest và checksum.
- [x] P2: đủ năm solver; baseline và optimized giữ trong mã.
- [x] P3: 18/18 unit tests PASS; log trong outputs/verification.
- [x] P4: 150 dòng sweep validation; 20 trường hợp test, 100 đầu ra solver, 240 dòng metric; benchmark trước/sau.
- [x] P5: báo cáo có ba phần bắt buộc, hình, gallery và hướng dẫn tái chạy; kiểm tra artifact/link PASS.
- [ ] Đối chiếu thực thi Fiji: chờ TV3 chạy plugin trên cùng TIFF, không thuộc bằng chứng Python đã hoàn tất.

Lần chạy nghiệm thu: [gallery](../outputs/bsds500_20261118_201351/gallery.html). [Báo cáo](04_bao_cao_phan_thanh_vien_2.md) chứa số đo và giới hạn.
