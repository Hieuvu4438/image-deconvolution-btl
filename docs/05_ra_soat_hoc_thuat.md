# Rà soát học thuật và kỹ thuật

## Tài liệu đối chiếu

Rà soát bản tổng hợp Markdown/DOCX ban đầu, bốn tài liệu thành viên trong kho, chương trình Python, sách chuyên khảo, bài báo gốc và mã nguồn DeconvolutionLab2. Báo cáo chính viết bằng tiếng Anh với bảng, hình lấy từ lần chạy `bsds500_20261118_234602`. Những phần minh họa có phương pháp và dữ liệu khác trong bản nháp được lưu làm tài liệu học tập.

| Điểm cần thống nhất | Kết luận và cách xử lý trong báo cáo |
|---|---|
| Mô hình suy giảm | Dùng hai biểu thức riêng cho nhiễu cộng và số đếm Poisson. Với quan sát đã chia cho $P$, phương sai là $b_i/P$. Mức 700 là kỳ vọng ở cường độ 1. Gaussian bị cắt âm để tất cả thuật toán nhận cùng đầu vào. |
| Tích chập, biên và tâm PSF | FFT trên lưới ảnh áp dụng tích chập tuần hoàn. Biên 0 tạo một toán tử khác. Với ảnh 256 × 256, xuất thêm PSF đồng kích thước và đúng tâm cho plugin. Ca xung bất đối xứng tại mép xác minh hướng, pha và quấn biên với sai số cực đại $2{,}90 \times 10^{-8}$. |
| Lọc ngược | Bản khảo sát plugin ghi NIF là chia thẳng $G/H$ và bỏ qua tham số. `Operations.divideStabilized` dùng chặn `max(epsilon, abs(H)**2)`. Báo cáo dẫn xuất phép nghịch đảo lý tưởng rồi trình bày NIF đã chặn, đặt ngưỡng 0,0001 ở cả hai nền tảng. |
| Tikhonov và Laplacian | TRIF dùng phạt chuẩn bình phương trực tiếp. RIF là cùng họ Tikhonov với $L$ khác. Python dùng stencil sai phân 5 điểm, Java dựng Laplacian phổ theo bình phương tần số góc trên lưới chẵn. So riêng hai biến thể, sau đó khớp công thức để kiểm chứng số. |
| RL và điều kiện dừng | Đưa $H^\top\mathbf{1}$ vào công thức tổng quát rồi rút gọn với PSF chuẩn hóa và biên tuần hoàn. Phân biệt hợp lý Poisson với PSNR theo ground truth. Dữ liệu validation Poisson cho PSNR giảm khi lặp tiếp từ 20 đến 80 bước ở bản Python. |
| Khác biệt RL và RLTV theo nền tảng | Plugin khởi tạo bằng ảnh quan sát và triệt tỉ số khi dự đoán dưới epsilon. Python dùng trung bình và cộng epsilon ở quy trình chính. Cách xử lý TV, gradient và điều kiện biên cũng khác. Đo chênh lệch bằng TIFF raw, kiểm chứng RL với bản khớp quy tắc. |
| Dữ liệu và ground truth | Các ca bo mạch, phantom 192 × 192 và tập BSDS500 trong bản nháp có điều kiện khác nhau. Đối chiếu chính dùng cùng 20 ca BSDS500. Ảnh xám trước suy giảm làm tham chiếu. Phantom hình học mới cung cấp ground truth độc lập với JPEG cho thực nghiệm phụ. |
| PSNR và SSIM | Thống nhất đỉnh 1, cửa sổ SSIM, căn ảnh, vùng chấm và chính sách raw/cắt dải. PSNR native của plugin dùng đỉnh thực của ảnh tham chiếu. Kiểm tra công thức quy đổi và tính các metric độc lập trên cùng TIFF trong Python và ImageJ. |
| Quy trình ImageJ | Đường dẫn macro và đầu ra của các ca cũ thiếu trong checkout ban đầu, nên các bảng đó không dùng làm bằng chứng đối chiếu. Thử đường chạy của plugin thực tế cho thấy `-stats no` mở cửa sổ ở bản JAR đã tải. Dùng `-stats false` và đầu ra `intact float noshow` trong hồ sơ mới. |
| Hiệu năng | Đo nhánh FFT đầy đủ so với nhánh real FFT và cache OTF bằng thời gian solver, có lượt làm nóng, bảy lần xen kẽ và kiểm sai khác toán học. Số đo bộ nhớ là lượng được `tracemalloc` theo dõi. Chỉ báo cáo so sánh hai bản Python. |
| Phạm vi kết luận | Xếp hạng giới hạn ở tập con, lưới tham số, mức nhiễu và cách đặt bài thử. Plugin nhận thông số chọn theo Python để kiểm chứng, nên kết quả không suy ra cấu hình tối ưu riêng của plugin. Đánh giá tách vùng biên và dùng SSIM, ảnh, đường cắt để đọc đánh đổi mà PSNR không phản ánh. |

## Điều chỉnh trong mã và bằng chứng

- Sửa biểu thức Laplacian rời rạc trong miền Fourier để dùng được cả ảnh có một hàng/cột. Thêm kiểm thử nghiệm hệ tuyến tính cho ca một hàng.
- Bổ sung chế độ đối chiếu RL theo quy tắc plugin, có kiểm thử tham chiếu tính trong miền không gian. Lưới RLTV quét đồng thời số bước và hệ số phạt.
- Khóa cấu hình trước test, dùng cùng đầu vào TIFF float32, tái tính đủ 240 bản ghi Python, kiểm tra 242 bản ghi ImageJ gồm cả ca xung. Bộ kiểm thử hiện tại có 20 ca đạt.
- Kết quả định lượng phụ gồm 720 bản ghi từ 6 kịch bản và 12 bản ghi phantom. Ảnh minh chứng trong PDF lấy từ TIFF hoặc được dựng trực tiếp từ dữ liệu thực nghiệm.
- Bổ sung dẫn xuất EM cho RL, phân tích độ lệch và khuếch đại nhiễu của Tikhonov, và đồ thị KL so với PSNR trên validation Poisson.
- Dùng thêm ground truth và PSF quang học của bộ EPFL Hollow Bars. So sánh tích chập 3D rồi chiếu với chiếu rồi tích chập 2D, sai khác tối đa khoảng `8.58e-8`. Giữ tổng của PSF float32 đã chuẩn hóa khi đưa vào Java bằng `-norm no`, sau khi kiểm tra thấy việc cộng lại PSF rộng trong Java tạo sai lệch số học ở các bộ lọc trực tiếp.

Tài liệu gốc dùng cho đối chiếu. [Sage và cộng sự, Methods, 2017](https://doi.org/10.1016/j.ymeth.2016.12.015), [mã nguồn DeconvolutionLab2](https://github.com/Biomedical-Imaging-Group/DeconvolutionLab2/tree/e9af0aba493ba137d70877154648e5583e376a81), [BSDS500 chính thức](https://www2.eecs.berkeley.edu/Research/Projects/CS/vision/grouping/resources.html) và [Wang và cộng sự, SSIM](https://doi.org/10.1109/TIP.2003.819861).
