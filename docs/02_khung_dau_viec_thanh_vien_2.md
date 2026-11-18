> **Bản nháp của nhóm.** Xem [báo cáo PDF đã thống nhất](../report/bao-cao.pdf) và [biên bản rà soát](05_ra_soat_hoc_thuat.md) để dùng cấu hình, lệnh và số liệu đã kiểm chứng.

# KHUNG ĐẦU VIỆC CỤ THỂ — THÀNH VIÊN 2 (CÀI ĐẶT, TỐI ƯU & KIỂM THỬ PYTHON)

- **Vai trò:** Thành viên 2 — Phạm Mạnh Dũng
- **Đầu vào tham chiếu:**
  - Lý thuyết & mô hình toán học của Thành viên 1 (Nguyễn Minh Hiếu): [01_bao_cao_nguyen_minh_hieu.md](01_bao_cao_nguyen_minh_hieu.md)
  - Khảo sát Fiji/ImageJ & quy chuẩn đối chiếu của Thành viên 3 (Nguyễn Thành Tâm): Tab `Nguyễn Thành Tâm` trong Google Docs chung
- **Mục tiêu cốt lõi:** Cài đặt lại đầy đủ các thuật toán Deconvolution từ công thức toán học bằng Python chuẩn hóa, tối ưu tốc độ/bộ nhớ, kiểm thử tự động (chứng minh đúng công thức tay và không lệch pha), và xuất dữ liệu/chỉ số đối sánh được 1-1 với **DeconvolutionLab2 (EPFL)**.

---

## 1. Phạm vi kỹ thuật & Yêu cầu chuyển hóa từ Báo cáo TV1 + TV3

### 1.1. Môi trường sẵn có trong workspace

- Python môi trường `.venv` (`pyproject.toml`): `numpy 2.5.2`, `scipy 1.18.1`, `opencv-python 5.0.0`, `Pillow 12.3.0`, `matplotlib 3.11.1`, `pandas 3.0.5`, thư viện chuẩn `unittest` và `tracemalloc`.
- **Nguyên tắc thư viện:** Có thể cài đặt trọn vẹn bằng **NumPy + SciPy + Pillow** (không cần thêm `scikit-image`, `tifffile` hay `pytest` để giữ môi trường gọn nhẹ; riêng `SSIM` tự viết đúng công thức (24) của Wang et al. 2004 trong báo cáo Hiếu).
- Dữ liệu DIP3E có sẵn ngay tại `../data/DIP3E_CH05_Original_Images/` (bao gồm `Fig0507(a)(ckt-board-orig).tif`).

### 1.2. Ánh xạ công thức Hiếu (TV1) → Module Python (TV2)

| Nhóm việc                 | Thành phần cần cài đặt                                                                                                                                                                                            | Tham chiếu báo cáo Hiếu                   | Lưu ý tương thích với DeconvolutionLab2 (TV3)                                                                                                                                            |
| :------------------------ | :---------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------- | :---------------------------------------- | :--------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------- | ---------------------------- | ------------------------------------------------------------------------------------------------------ | ---------------------------- | ------------------------------------------------------------------------------- |
| **1. Tiền xử lý & PSF**   | Sinh PSF Gaussian 2D chuẩn hóa ($\sum h = 1$), hàm căn tâm `psf2otf` (`ifftshift` / `roll`), kiểm tra tổng PSF, xử lý biên (`circular` / `reflect` / `zero-pad`), đọc/ghi TIFF 32-bit không nén                   | Chương 1, Mục 2.1, 2.7, Công thức (1)–(4) | **Bắt buộc** hỗ trợ PSF kích thước chẵn tâm `(nx//2, ny//2)` (VD: `16×16` tâm `(8,8)`, `18×18` tâm `(9,9)`) và lưu TIFF 32-bit float **không nén** (`compression=None` / raw qua Pillow) |
| **2. Mô phỏng suy giảm**  | Bài toán thuận $g = f * h + n$ bằng FFT; sinh nhiễu cộng Gaussian ($\sigma_n$) và nhiễu Poisson (quy đổi theo mức photon, VD: 700 photon/mức 1); tái tạo bộ ảnh tổng hợp `192 × 192`                              | Mục 1.3, 1.4, 4.1, Công thức (2), (7)     | Tái lập chính xác 2 bài test chuẩn: `Fig0507` ($\sigma_{\text{psf}}=1.5, \sigma_n=3.0$, seed `42`) và `synthetic_192` ($\sigma_{\text{psf}}=1.9$, Poisson 700, seed `20261117`)          |
| **3. Thuật toán NIF**     | Lọc ngược trực tiếp miền Fourier $\widehat{F} = G / H$, bảo vệ chia 0 bằng ngưỡng $                                                                                                                               | H                                         | ^2 < \varepsilon$                                                                                                                                                                        | Mục 3.2, Công thức (8), (9)  | Đối chiếu hành vi khuếch đại nhiễu khi không điều chuẩn                                                |
| **4. Thuật toán TRIF**    | Tikhonov bậc 0 trong miền Fourier: $\widehat{F} = \frac{H^\* G}{                                                                                                                                                  | H                                         | ^2 + \lambda}$                                                                                                                                                                           | Mục 3.4, Công thức (11)–(14) | Phải nhận tham số $\lambda > 0$ đồng nhất với `TRIF <λ>` của DeconvolutionLab2                         |
| **5. Thuật toán RIF**     | Điều chuẩn Laplacian miền Fourier: $\widehat{F} = \frac{H^\* G}{                                                                                                                                                  | H                                         | ^2 + \lambda                                                                                                                                                                             | L\_{\text{fft}}              | ^2}$ với $L = \begin{bmatrix} 0 & 1 & 0 \\ 1 & -4 & 1 \\ 0 & 1 & 0 \end{bmatrix}$                      | Mục 3.5, Công thức (15)–(17) | Lưu ý cách đặt tâm kernel Laplacian `3×3` vào FFT (`psf2otf`) để không lệch pha |
| **6. Thuật toán RL**      | Cập nhật lặp Richardson–Lucy không âm $f^{(k+1)} = f^{(k)} \odot \left[h^\star * \frac{g}{h * f^{(k)} + \varepsilon}\right]$, hỗ trợ ảnh khởi tạo (phẳng hoặc $g$), dừng theo số vòng $K$ hoặc sai khác tương đối | Mục 3.6, Phụ lục B.1, Công thức (18)–(20) | Tái sử dụng `rfft2(psf)` và `conj(rfft2(psf))` tính sẵn trước vòng lặp để tối ưu tốc độ & RAM                                                                                            |
| **7. Thuật toán RLTV**    | Richardson–Lucy kết hợp phạt Total Variation (TV) theo công thức (21): nhân thêm hệ số điều chỉnh dựa trên độ phân kỳ gradient chuẩn hóa $\text{div}\left(\frac{\nabla f}{\sqrt{                                  | \nabla f                                  | ^2 + \delta^2}}\right)$                                                                                                                                                                  | Mục 3.7, Công thức (21)      | Hỗ trợ cặp tham số `(K, λ, δ)`, bảo toàn $f \ge 0$, ghi rõ phiên bản cập nhật TV đã dùng trong báo cáo |
| **8. Chỉ số & Profiling** | `compute_mse`, `compute_psnr`, `compute_ssim` (cửa sổ Gaussian $\sigma=1.5$, $C_1=(0.01 \cdot MAX)^2, C_2=(0.03 \cdot MAX)^2$), đo `wall_time_s` (`time.perf_counter`) và `peak_ram_mb` (`tracemalloc`)           | Mục 4.2, Công thức (22)–(24)              | Cho phép đánh giá cả thang `[0, 1]` và `[0, 255]`, có tùy chọn `clip` rõ ràng (ghi nhận lưu ý ở Mục 4.5 của Hiếu)                                                                        |

---

## 2. Cấu trúc thư mục mã nguồn đề xuất cho Thành viên 2

Đề xuất tổ chức gọn trong `team-project/code/` (sử dụng đường dẫn tương đối hoàn toàn):

```text
team-project/
├── docs/
│   ├── 01_bao_cao_nguyen_minh_hieu.md       # Báo cáo lý thuyết TV1 (đã lưu)
│   ├── 02_khung_dau_viec_thanh_vien_2.md    # Tài liệu khung đầu việc (file này)
│   ├── 03_ke_hoach_trien_khai_hom_nay.md    # Lịch trình thực thi trong ngày
│   └── 04_bao_cao_phan_thanh_vien_2.md    # Báo cáo phần Thành viên 2 (sẽ viết)
├── assets/hieu/                             # 12 hình minh họa từ báo cáo Hiếu
└── code/                                    # Không gian làm việc chính của TV2
    ├── deconv/
    │   ├── __init__.py
    │   ├── io_utils.py                  # Đọc/ghi ảnh, TIFF 32-bit raw, chuẩn hóa [0,1] / [0,255]
    │   ├── psf.py                       # Sinh PSF Gaussian, kiểm tra chuẩn hóa, psf2otf (chẵn/lẻ)
    │   ├── degradation.py               # Convolution FFT, nhiễu Gaussian/Poisson, sinh ảnh 192×192
    │   ├── solvers.py                   # NIF, TRIF, RIF, RL, RLTV (+ trace lịch sử lặp)
    │   ├── metrics.py                   # MSE, PSNR, SSIM (Wang et al., 2004)
    │   └── benchmark.py                 # Wrapper đo thời gian + Peak RAM (tracemalloc)
    ├── tests/
    │   ├── test_hand_examples.py        # Kiểm chứng 100% các ví dụ số trong báo cáo Hiếu
    │   ├── test_properties.py           # Kiểm chứng 0-pixel shift, sum(PSF)=1, RL >= 0, TIFF roundtrip
    │   └── test_solvers.py              # Kiểm chứng hội tụ & ổn định các thuật toán trên ảnh thật
    ├── scripts/
    │   ├── run_experiments.py           # Chạy trọn bộ Thí nghiệm 1 (Fig0507) & Thí nghiệm 2 (192×192)
    │   └── export_report_artifacts.py   # Xuất CSV đối chiếu, biểu đồ & log cho Chương 4 báo cáo
    └── outputs/                         # Ảnh khôi phục (.tif/.png) và bảng CSV kết quả của TV2
```

---

## 3. Phân rã 6 nhóm đầu việc cụ thể & Tiêu chí nghiệm thu (Definition of Done)

### Nhóm việc A — Khởi tạo & Chuẩn hóa I/O, PSF, Mô hình suy giảm

- [ ] **A1.** Viết `io_utils.py`: đọc ảnh xám chuẩn hóa `float64` (hỗ trợ cả `[0, 1]` và `[0, 255]`), ghi TIFF 32-bit float **không nén** bằng Pillow để `DeconvolutionLab_2.jar` đọc trực tiếp không lỗi.
- [ ] **A2.** Viết `psf.py`:
  - `gaussian_psf(size, sigma)` đảm bảo $\sum h = 1.0$.
  - `psf_to_otf(psf, out_shape)` tự động tính đúng tâm mảng cho cả `size` lẻ (`size // 2`) và `size` chẵn (`size // 2`, khớp quy ước `AcademicFFT` của DeconvolutionLab2).
- [ ] **A3.** Viết `degradation.py`:
  - `forward_blur_fft(img, psf)` và thêm nhiễu Gaussian ($\sigma_n$) / Poisson (`photon_peak`, ví dụ `700`).
  - Hàm `generate_synthetic_test(size=192)` tái tạo ảnh tổng hợp gồm điểm sáng, đường thẳng, thanh sáng và vòng tròn tương đồng Chương 4 của Hiếu.
- **Tiêu chí xong:** Chạy test impulse tại `(32, 32)` trên lưới `64×64` cho kết quả sau khôi phục có cực đại nằm đúng tại `(32, 32)` (độ dịch = `0` pixel).

### Nhóm việc B — Cài đặt 5 thuật toán Deconvolution (NIF, TRIF, RIF, RL, RLTV)

- [ ] **B1.** `naive_inverse_filter(g, psf, eps=1e-4)` — chia miền Fourier có ngưỡng bảo vệ mẫu số $|H|^2$.
- [ ] **B2.** `tikhonov_rif(g, psf, lam)` (**TRIF**) — công thức (13) bằng `rfft2`/`irfft2`.
- [ ] **B3.** `laplacian_rif(g, psf, lam)` (**RIF**) — công thức (15)–(17) với kernel Laplacian 3×3 đưa về cùng `out_shape`.
- [ ] **B4.** `richardson_lucy(g, psf, num_iter, init_mode="mean", eps=1e-12, return_history=False)` (**RL**) — công thức (18), hỗ trợ theo dõi PSNR/sai khác qua từng vòng lặp.
- [ ] **B5.** `richardson_lucy_tv(g, psf, num_iter, lam, delta=1e-6)` (**RLTV**) — công thức (21), cập nhật nhân có nhân tử điều chuẩn TV $\frac{1}{1 - \lambda \, \text{div}(\nabla f / \sqrt{|\nabla f|^2 + \delta^2})}$, kèm bảo vệ tránh chia 0 và giữ $f \ge 0$.
- **Tiêu chí xong:** Tất cả 5 hàm có type hints đầy đủ, docstring rõ ràng công thức tham chiếu, nhận `np.ndarray` 2D bất kỳ kích thước (kể cả không vuông như `448×464`).

### Nhóm việc C — Tối ưu hóa chương trình (Optimization)

- [ ] **C1. Dùng `rfft2` / `irfft2` thay cho `fft2` / `ifft2`:** Vì ảnh và PSF là thực, khai thác đối xứng Hermitian giảm ~50% dung lượng phổ tần số và tăng tốc ~1.5–2×.
- [ ] **C2. Tiền tính toán (Precomputation) ngoài vòng lặp RL/RLTV:** Tính `H = np.fft.rfft2(...)` và `H_conj = np.conj(H)` đúng 1 lần trước vòng `for`, không gọi lại `fft` cho PSF mỗi vòng lặp.
- [ ] **C3. Vector hóa hoàn toàn:** Không dùng vòng lặp Python theo pixel cho tích chập, Laplacian hay tính gradient TV (`np.roll` / slicing NumPy).
- [ ] **C4. Quản lý bộ nhớ & `dtype`:** Cho phép cấu hình `float32` vs `float64` khi chạy ảnh lớn, giải phóng mảng trung gian không cần thiết trong vòng lặp RL để tối ưu Peak RAM.
- **Tiêu chí xong:** Đo `wall_time_s` và `peak_ram_mb` trên `Fig0507 (448×464)` cho thấy tốc độ vượt trội hoặc tương đương `DeconvolutionLab_2.jar` (TRIF/RIF < 0.1s, RL 50 vòng < 0.5s).

### Nhóm việc D — Kiểm thử tự động (Unit Tests & Regression Tests)

- [ ] **D1.** `test_hand_examples.py` kiểm chứng chính xác bằng `assertAlmostEqual` tất cả ví dụ số trong báo cáo Hiếu:
  - Convolution 1D: `[0, 0, 8, 0, 0] * [0.25, 0.5, 0.25] == [0, 2, 4, 2, 0]` và bài tập `[0, 0, 12, 0, 0] -> [0, 3, 6, 3, 0]`.
  - Convolution 2D: ảnh `5×5` tâm `9` qua kernel `3×3` toàn `1/9` -> `9` ô tâm bằng `1`.
  - NIF / ma trận 2×2: `H = [[0.75, 0.25], [0.25, 0.75]]`, chứng minh khuếch đại nhiễu ra `[8.2, -0.2]`.
  - Tikhonov 1D: `H=0.1, g=0.45` với `λ ∈ {0, 0.001, 0.01}` ra `{4.5, 4.0909, 2.25}`.
  - RL 2 pixel qua 2 vòng lặp: $f^{(0)} = [4, 4] \to f^{(1)} = [5, 3] \to f^{(2)} \approx [5.7143, 2.2857]$.
  - Laplacian `L`: vùng phẳng `5` ra `0`, tâm `9` lân cận `5` ra `-16`.
- [ ] **D2.** `test_properties.py` kiểm chứng bất biến: tổng PSF = `1.0`, bảo toàn tổng cường độ khi làm mờ không cắt biên, không dịch tâm với PSF chẵn & lẻ, đầu ra RL luôn `>= 0`, `NIF` trên ảnh không nhiễu khôi phục PSNR > `60 dB`.
- **Tiêu chí xong:** `.\.venv\Scripts\python.exe -m unittest discover -s team-project/code/tests -v` PASS 100% không skip/lỗi.

### Nhóm việc E — Thực nghiệm đối chứng & Đo lường hiệu năng

- [ ] **E1.** Cài đặt `metrics.py` (`MSE`, `PSNR`, `SSIM` chuẩn Wang et al. 2004).
- [ ] **E2.** Chạy **Thí nghiệm 1 (`Fig0507(a)(ckt-board-orig).tif`)** với cùng cấu hình của TV3 (`σ_psf=1.5`, PSF `16×16`, `σ_n=3.0`, seed `42`), xuất bảng `exp1_python_metrics.csv`.
- [ ] **E3.** Chạy **Thí nghiệm 2 (`synthetic_192`)** với cấu hình Chương 4 Hiếu (`σ_psf=1.9`, Poisson 700 photon, seed `20261117`), xuất bảng `exp2_python_metrics.csv`.
- [ ] **E4.** Quét tham số (`λ` cho TRIF/RIF và `K` cho RL) phục vụ biểu đồ tối ưu và đối chiếu trực tiếp với `DeconvolutionLab_2.jar` (có thể chạy trực tiếp bằng `java` có sẵn trên máy nếu muốn verify chéo ngay).

### Nhóm việc F — Đóng góp nội dung báo cáo (~6 trang) & Slide cho nhóm

- [ ] **F1.** Viết file Markdown phần báo cáo của Thành viên 2 (`team-project/docs/04_bao_cao_phan_thanh_vien_2.md`) **bắt buộc theo đúng 3 phần cốt lõi** (có thể bổ sung phần phụ trợ nếu cần):
  1. **Phần 1 — Cài đặt thuật toán:** Kiến trúc module Python, luồng xử lý I/O & căn tâm PSF, ánh xạ chi tiết công thức toán học (TV1) sang mã nguồn cho cả 5 thuật toán (`NIF`, `TRIF`, `RIF`, `RL`, `RLTV`).
  2. **Phần 2 — Tối ưu chương trình:** Dùng `rfft2`/`irfft2`, tiền tính OTF ngoài vòng lặp, vector hóa hoàn toàn bằng NumPy, quản lý kiểu dữ liệu/`Peak RAM`, kèm bảng benchmark trước–sau tối ưu (thời gian chạy & bộ nhớ).
  3. **Phần 3 — Kiểm thử chương trình:** Hệ thống Unit Test kiểm chứng 100% ví dụ tính tay của TV1, kiểm chứng bất biến vật lý (tổng PSF = 1, 0-pixel shift, $f \ge 0$, TIFF 32-bit raw), và kết quả chạy test PASS.
  4. _(Phần bổ sung nếu cần)_ **Thực nghiệm đối chứng & Tích hợp nhóm:** Bảng số liệu định lượng (`MSE`, `PSNR`, `SSIM`) trên `Fig0507` và `synthetic_192`, đối chiếu với `DeconvolutionLab2` (TV3) và giao diện đầu ra cho TV4.
- [ ] **F2.** Xuất bộ hình ảnh minh họa rõ nét (input vs output, zoom ROI, biểu đồ benchmark & PSNR/SSIM) vào `outputs/` để ghép vào báo cáo 30 trang và 10 slide thuyết trình.
- [ ] **F3.** Đồng bộ toàn bộ nội dung 3 phần bắt buộc (+ phần bổ sung nếu có) lên tab `Phạm Mạnh Dũng` trong Google Docs chung của nhóm.
