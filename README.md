# Bài tập lớn Xử lý ảnh số — Khôi phục ảnh bằng deconvolution

**Nhóm:** cần bổ sung mã nhóm trước khi nộp. **Nền tảng tham chiếu:** Fiji/ImageJ với DeconvolutionLab2. **Phần Python:** cài đặt và thực nghiệm trên ảnh BSDS500 của Berkeley.

Bài tập nghiên cứu mô hình ảnh mờ $g=f*h+n$, cài đặt các thuật toán khử mờ và đánh giá bằng PSNR/SSIM. Ảnh BSDS500 được dùng làm ảnh tham chiếu; nhóm tự tạo ảnh mờ và nhiễu với PSF đã biết. Nhãn phân đoạn của BSDS500 không phải ground truth cho bài khử mờ. Kết quả hiện có là **thực nghiệm Python**; phần đối chiếu thực chạy trong Fiji cần bổ sung bằng ảnh xuất và log của plugin.

## Đọc từ đâu

| Tài liệu | Nội dung |
|---|---|
| [Báo cáo lý thuyết](docs/01_bao_cao_nguyen_minh_hieu.md) | PSF, convolution, NIF, TRIF, RIF, RL, TV và các ví dụ tính tay |
| [Khung đầu việc thành viên 2](docs/02_khung_dau_viec_thanh_vien_2.md) | Phân công ban đầu; các chỉ tiêu DIP3E/tốc độ trong file này đã được thay bằng kế hoạch BSDS500 |
| [Kế hoạch BSDS500](docs/03_ke_hoach_trien_khai_hom_nay.md) | P0–P5, đầu ra và gate kiểm chứng từng phase |
| [Báo cáo thực nghiệm thành viên 2](docs/04_bao_cao_phan_thanh_vien_2.md) | Ba phần bắt buộc: cài đặt, tối ưu trước–sau, kiểm thử; bảng số liệu và giới hạn |
| [Hướng dẫn mã nguồn](code/README.md) | Lệnh chạy, cấu trúc artifact, quy ước metric và bàn giao Fiji |

Thành viên 1 — Nguyễn Minh Hiếu: lý thuyết và mô hình toán học. Thành viên 2 — Phạm Mạnh Dũng: cài đặt, tối ưu, kiểm thử và thí nghiệm Python. Thành viên 3 — Nguyễn Thành Tâm: workflow Fiji/DeconvolutionLab2 và đối chiếu đầu ra. Thành viên 4: chuẩn bị dataset, ground truth và phân tích metric theo phân công nhóm; tên và đóng góp thực tế cần được nhóm xác nhận khi ghép báo cáo cuối.

## Chạy lại phần Python

Chạy từ **gốc repository** trên Windows/PowerShell. Dùng `.venv` và dependency hiện có trong `pyproject.toml`; các lệnh dưới đây không cài thêm gói.

```powershell
.\.venv\Scripts\python.exe team-project/code/scripts/prepare_data.py
.\.venv\Scripts\python.exe team-project/code/scripts/check.py
.\.venv\Scripts\python.exe team-project/code/scripts/run_experiments.py
.\.venv\Scripts\python.exe team-project/code/scripts/audit_artifacts.py
.\.venv\Scripts\python.exe team-project/code/scripts/build_report.py
```

`prepare_data.py` tải archive từ [trang BSDS500 chính thức của Berkeley](https://www2.eecs.berkeley.edu/Research/Projects/CS/vision/grouping/resources.html), chọn 5 ảnh validation và 10 ảnh test theo tên tăng dần rồi lưu SHA256. Lần đầu cần mạng để tải khoảng 70,8 MB. `run_experiments.py` tạo một thư mục mới cho mỗi lần chạy; `team-project/outputs/LATEST.txt` chỉ đến lần chạy gần nhất. `build_report.py` lấy số đo từ thư mục này để cập nhật báo cáo sau khi `audit_artifacts.py` PASS. Nếu chỉ muốn đọc kết quả hiện có, xem trực tiếp báo cáo thực nghiệm, không cần chạy lại.

Các thuật toán nằm trong `code/deconv/solvers.py`: NIF, TRIF, RIF, RL và RLTV. Bản baseline dùng FFT phức đầy đủ; bản tối ưu dùng real FFT và cache OTF. Unit test kiểm tra ví dụ tính tay trong báo cáo Hiếu, đối chiếu toán tử độc lập, I/O, metric và tính tương đương hai bản. [File kiểm thử](code/tests/test_deconv.py) và log thực chạy trong `outputs/verification/` lưu bằng chứng kiểm chứng.

## Kết quả Python đã ghi nhận

Lần chạy nghiệm thu `bsds500_20261118_201351` có **18/18 unit test PASS**. Trên 10 ảnh test với hai loại nhiễu, pipeline lưu 100 ảnh khôi phục và 240 bản ghi metric raw/clipped. Audit tái tính metric từ TIFF đã lưu khớp CSV trong ngưỡng `1e-4`. Trên một ảnh validation 256×256, bản tối ưu nhanh hơn baseline khoảng **2,05–3,09 lần** tùy thuật toán; đây là số đo trên môi trường đã ghi trong artifact, không phải so tốc độ với Java/Fiji.

Ở những cấu hình được chọn trên 5 ảnh validation, RL đạt PSNR trung bình cao nhất trên subset Gaussian test; RLTV đạt cao nhất trên subset Poisson test. Bảng PSNR/SSIM, cấu hình và giới hạn diễn giải nằm trong [báo cáo TV2](docs/04_bao_cao_phan_thanh_vien_2.md).

Ảnh TIFF raw, PNG, manifest, CSV và [gallery của lần chạy nghiệm thu](outputs/bsds500_20261118_201351/gallery.html) nằm trong `outputs/`. `data/` và `outputs/` được bỏ qua trong Git vì kích thước lớn và có thể tái tạo. Vì vậy, các liên kết từ báo cáo đến hình/CSV cục bộ chỉ hoạt động ở checkout đã chạy pipeline hoặc được bàn giao kèm thư mục `outputs/`; bảng số liệu chính vẫn đọc được trong báo cáo.

## Cấu trúc chính

```text
team-project/
├── README.md
├── .gitignore
├── docs/                 # lý thuyết, kế hoạch và báo cáo TV2
├── assets/hieu/          # hình cho báo cáo lý thuyết
├── code/
│   ├── deconv/           # I/O, PSF, solver và metric
│   ├── tests/            # unittest
│   └── scripts/          # tải dữ liệu, chạy, audit, sinh báo cáo
├── data/bsds500/         # tải từ Berkeley; Git bỏ qua
└── outputs/              # ảnh, CSV, log, gallery; Git bỏ qua
```

Để hoàn tất phần Fiji, dùng cùng `reference.tif`, `observed.tif`, `psf.tif` và cấu hình đã khóa trong run. Thành viên 3 cần lưu đầu ra thực chạy, tham số plugin, log và ảnh chụp giao diện trước khi thêm bảng so sánh Python–Fiji. Các phép thử hiện tại là mô phỏng 2D với PSF biết trước trên subset BSDS500, chưa phải benchmark phân đoạn chính thức hay kiểm chứng trên ảnh kính hiển vi thực.
